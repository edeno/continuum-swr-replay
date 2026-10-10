# Inventory comparison: seven detectors over one substrate

*Design spec. Status: approved; the accounting — schema, availability, overlap, components,
fragment-merge recount, cells and SWR label matrix — is implemented in
`src/continuum_swr_replay/analysis/inventory_comparison/` and tested on synthetic
inventories. Not yet run on recorded data: the per-detector adapters that build coverage
from recordings, the runner, and the sampled examples remain, and excluded time is not
yet reported by reason (the observation tables carry a reason per stretch; nothing
aggregates it per comparison).*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Depends on [01](01-yu-swr-detector.md) and [02](02-memoryless-sequence.md). Covers how the
seven event inventories are brought onto a common schema and how their overlap is counted.
What *differs* about the events once matched is [04](04-event-features.md).

## Common schema

Each inventory produces one table, one row per event, over the same epochs and the same
speed-eligible population (overview X4). Every table carries:

- **Identity** — `animal`, `day`, `epoch`, `brain_area`, `detector`, `event_index`,
  and a key, `event_key` (`"<detector>:<animal><day>_<epoch>:<native key>"`), unique
  across every inventory and epoch compared (overview X8): SWR and MUA native keys
  restart in each epoch, so a key without the epoch would join events that never met.
- **Timing (seconds)** — `event_start_s`, `event_end_s`, `interval_span_s`,
  `event_midpoint_s`. `interval_span_s` is defined identically everywhere as
  `event_end_s - event_start_s`, because the native duration conventions differ: our
  canonical cohort's duration includes a sample-width convention that the sequence
  detector's first-to-last-midpoint duration does not. Native duration fields are
  retained alongside it, never substituted for it.
- **Speed** — `animal_max_speed_cm_s`, `animal_mean_speed_cm_s`,
  `whole_event_speed_eligible`, named as the canonical classified table names them.
- **Provenance** — `run_id` joining to the frozen configuration and artifact manifest.

Detector-specific statistics stay in detector-specific columns, namespaced by detector,
and are never compared across detectors as if they meant the same thing.

**No inventory is a truth set.** The word "detected" is always qualified by which detector.

## Observation availability

Each inventory also exports an observation-interval table, keyed by `animal`, `day`,
`epoch`, `brain_area`, `detector`, and `run_id`, with interval bounds, availability,
and an exclusion reason where unavailable. It partitions the epoch into observed and
unavailable intervals independently of whether any events were detected. A successfully
evaluated interval with no detections is observed; a missing artifact or failed run is
unavailable. Zero spikes during a verified recording interval are valid observations.

Construct availability from the required recordings, timestamps, position/speed, and
model outputs. Split at both non-finite samples and timestamp gaps; do not interpolate
availability across missing data. Include each detector's required input support: an
80 ms window is evaluable at its midpoint only if its full support is available, so
window edges and gaps can reduce observation time. A failed amplitude, probability,
sequence, or significance criterion is a detection outcome, not missing coverage.
Record the source masks and timing conventions in the manifest.

The reader of the observation tables (`availability.coverage_from_observations`) first
removes excluded time (`swr_source.excluded_intervals`: the artifact mask widened by
0.3 s) from every detector's observed intervals, since no detector's events there are
analysed; the tables themselves record what each detector read. Intersect each
detector's observed intervals with the shared finite-speed < 4 cm/s eligibility mask
to obtain its comparison exposure, `E_d`. A comparison involving a
set of detectors uses the intersection of their exposures. Pairwise summaries use
`E_A ∩ E_B`; seven-detector exclusivity and no-detector time use all seven exposures.
Report the exposure duration and excluded time by reason for each comparison. Empty
shared exposure produces an unavailable summary, not a zero rate or zero overlap.

For event-count denominators, require the source event's entire native interval to lie
within the shared exposure. Retain excluded events with an availability flag; do not
clip, split, or re-detect them to qualify. A confirmed overlap can still use a valid
counterpart event that extends beyond the shared exposure, since that event was
observed; it need not qualify for the reverse direction's denominator. Every
inventory is first restricted to its speed-eligible events, so no counterpart spans
movement: a movement-spanning SWR cannot mark a still sequence, as the canonical
SWR labels are speed-gated before they mark a non-local event. Cached events
that bridge their own detector's unavailable data are set aside into their own table
before comparison (`remove_bridging_events`), and their full spans are marked
unavailable pending corrected detection, so discarding a corrupt event cannot create
an apparent non-detection. The memoryless inventory is one named cohort of that
detector's candidate table (`published`, or `replay_criteria` before the significance
test), never every candidate row, since that table keeps the detector's rejections. Clip event
unions to shared exposure only for **time coverage**.

## Matching and counting

Build a table of **every** strict positive-duration overlap pair between every pair of
inventories, carrying both keys, intersection duration, temporal
intersection-over-union, the fraction of each event covered by the other, and signed
onset/offset differences. Pair counts describe the matching relation; they are not
independent event counts.

`event_strict_overlap_keys` can check membership using timebin labels but returns
neither pairs nor overlap lengths, so pairing operates on the canonical intervals.

**Denominators are per-detector.** Report, for each ordered pair of detectors, the
proportion of detector A's events overlapping any event of detector B, and the reverse.
Each fully observed source event contributes once to its direction's denominator
regardless of how many matches it has. Keep availability flags on the pair table and
report excluded event counts. Show overlap-length and IoU distributions so a one-sample
edge contact is not counted as agreement with a fully coincident event.

**Fragment-merge recount.** Our detector can split one replay into several events when
the non-local probability dips below threshold. Provide a recount in which our events
separated by less than a stated gap (a fixed grid: 0, 10, 20, 50 ms) are merged before
overlap is computed, alongside the native count. The gap is end-to-start — from the
last sample of one event to the first of the next — counted in whole samples, so a gap
of exactly a grid value never merges. The native count remains primary; the
recount diagnoses how much count differences depend on segmentation (04). It uses
existing intervals and requires no re-decoding. Keep merges within an epoch and a
continuous eligible immobility bout, checking speed throughout each intervening gap;
do not bridge movement or missing samples. These are secondary accounting intervals,
not a replacement detected cohort.

**Also report time coverage** of the intersection and union of each pair of cohorts.
Express fractions relative to that pair's shared exposure, with its duration shown.
Coverage is less sensitive to segmentation than event counts, and the two can disagree —
one detector splitting an event into three does not change the time it accounts for.

**Overlap is not transitive.** A can overlap B and B overlap C without A overlapping C.
Membership in "all seven" is therefore not well-defined by pairwise overlap alone.
Summarize connected components of the multipartite overlap graph, labeled by which
detectors contribute to each component, and report component size and span as
diagnostics — a long chain can fuse events that do not individually overlap.
**Do not turn components into replacement event intervals.**

### The cells that pairwise counting misses

The cells of most interest are the ones no pairwise table reaches:

- **Events found by exactly one detector**, per detector.
- **SWRs matched by neither decoding-based detector** — absence of a detected match,
  not proof that the ripple has no spatial content. Characterize their activity with
  the same diagnostics as every other inventory. This is computable because SWR
  inventories are first-class here rather than labels attached to someone else's events.
- **Sequences matched by no LFP- or rate-triggered inventory** — events the standard
  field pipelines would not have found at all. This is the neither-ripple-nor-burst
  cell, distinct from ripple-less (overview): an MUA-matched, SWR-unmatched sequence is
  ripple-less but not in this cell.
- **Time accounted for by no detector**, with total shared exposure as its denominator.

These cells require observation by every detector whose absence is asserted. Incomplete
coverage yields unknown membership, not an unmatched or no-detector result; report it
separately. No-detector time is shared exposure minus the union of detected intervals.

## The SWR label matrix

Distinct from the inventory comparison above, and the reason level 3 needs no separate
sweep (overview X6). Each *sequence* cohort — ours and the memoryless one — is labeled
under each of the four SWR definitions (Kay, Karlsson, Yu and Zugaro), giving a
cohort × SWR-definition matrix of non-SWR fractions. The canonical Spyglass `SWR_event`
that the existing tables use is not a fifth column of the matrix: it is the version check
for the direct Kay inventory, which is the matrix's Kay column
([refresh design](../2026-10-03-full-pipeline-refresh-design.md), decision R4).

Labels are `SWR`, `non-SWR`, or `unknown`. Incomplete observation over a sequence's
native interval yields `unknown`, with any observed overlap retained in the pair table
for diagnostics. For the matrix, restrict each sequence cohort to events fully within
its exposure intersected with all four SWR exposures, so all four fractions use the
same events. Report the eligible denominator and exclusions alongside the fractions;
unknown labels never count as non-SWR.

A non-SWR fraction is a function of the SWR definition as much as of the events, so it is
never reported without its definition attached. Where the four definitions disagree
about a given event, that disagreement is itself reported: an event labeled ripple-less
by one algorithm and ripple-bearing by another describes dependence on the detection
rule; it does not establish which label is correct.

Keep a separate table of paired label agreement. A matched pair's two events can carry
different SWR labels purely because their bounds differ.

## Deferred eligibility and encoding controls

Neither is an initial deliverable. Reopen a control when a particular disagreement
requires it, label the analysis as secondary, and retain the primary cohorts and results.

1. **Native speed gates** (overview X4). Re-run each method under its own published
   gate — Widloski's per-bin < 5 cm/s, ours whole-event < 4 cm/s, Yu's < 4 cm/s — and
   report how much of each cohort difference the gate accounts for.
2. **Running-only encoding** (overview X5). Refit the clusterless encoding model on
   running periods only and re-run **both** decoding-based methods on it. Applied to both
   or neither; applying it to one would make the encoding model a free variable
   distinguishing them.
3. **Reference-channel ripple power** (deferred). The vandermeerlab workflow thresholds
   the z-scored difference between the ripple channel's ripple-band power and a control
   channel's, and the consensus statement (Liu et al. 2022) recommends difference or
   re-referenced derivations to remove far-field artifacts. Artifact handling is excluded
   from the detector package by decision, but the corpus-callosum reference tetrodes make
   the derivation available here; reopen it only if artifact-driven SWR events are
   suspected in a specific disagreement.

Threshold robustness has a separate staged plan in [04](04-event-features.md): first
complete the default inventories and report, then measure costs on a small pilot before
choosing a broader scope. The SWR-label matrix above uses the retained settings and does
not substitute for threshold robustness.

## Reporting

All main summaries show per-session values and animal-level consistency, with sessions
nested within animal; pooled event summaries are descriptive. Report native duration
distributions before any duration-stratified comparison, since longer intervals have more
opportunity to overlap anything. Report native counts alongside comparison-eligible
counts, missing features, and skipped epochs. Rates use counts qualified for their
corresponding observation exposure; availability within epochs also defines the paired
population.

## Testing

| test | what it protects |
|---|---|
| One-to-many and many-to-many overlaps count each event once per detector denominator | The denominators, which are the easiest thing to get silently wrong |
| Matched events may carry different SWR labels; the paired-label table records it rather than resolving it | Boundary-driven label disagreement stays visible |
| Disjoint cohorts, empty cohorts, and single-event cohorts | Degenerate inputs |
| Connected components: a chain A–B–C where A and C do not overlap is reported as one component with its span, and is not collapsed into an interval | Non-transitivity |
| Coverage and count summaries disagree as expected on a constructed split/merge case | That both are reported because they answer different questions |
| Fragment-merge recount at gap 0 reproduces native counts; a large gap merges all events within each continuous eligible bout but never across movement, missing samples, or epochs | The recount changes segmentation without changing eligibility |
| `interval_span_s` is computed identically for every inventory, and native duration fields are never substituted for it | The convention mismatch between cohorts |
| An event at an epoch boundary is not matched across epochs | Session scoping |
| Empty inventories carry the schema's dtypes, not object, so concatenation across epochs preserves column types | The canonical builder's empty return is all-object |
| A sequence during missing LFP has unknown SWR status and exclusivity; NaN runs and missing timestamps both remove coverage | Missing observations cannot masquerade as non-detection |
| Complete data with zero detections remains observed; missing outputs and zero shared exposure yield unavailable summaries | Empty inventories differ from failed or unobserved runs |
| An event crossing a coverage boundary is excluded from its count denominator without changing its bounds; a valid counterpart crossing that boundary still supplies an overlap for an eligible source event | Directional denominators and native timing remain consistent |
| An 80 ms window overlapping a data gap has unavailable midpoint coverage; no-detector time and baseline windows exclude it | Observation support includes the detector's input window |
| The SWR matrix uses the same eligible sequence events across its four columns and excludes unknown labels | Definition changes are not confounded with differing availability |

## Deliverables

1. The seven inventories and their observation-interval tables, with native and eligible
   counts, exposure durations, rates, and exclusions.
2. The all-pairs overlap table and the per-detector denominators.
3. The single-detector and no-detector cells, including SWRs with no detected sequence and
   sequences no triggered pipeline found.
4. The cohort × SWR-definition non-SWR matrix, with paired label disagreement.
5. Reproducibly sampled examples from each cell, on a shared time axis showing spikes,
   both posterior representations, gate masks, speed, and LFP — with the sampling rule
   stated and counterexamples included in both directions.

The eligibility and encoding controls are conditional follow-ups, not prerequisites
for these deliverables.
