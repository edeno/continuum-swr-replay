# Inventory comparison: six detectors over one substrate

*Design spec. Status: approved, not yet implemented.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Depends on [01](01-yu-swr-detector.md) and [02](02-memoryless-sequence.md). Covers how the
six event inventories are brought onto a common schema and how their overlap is counted.
What *differs* about the events once matched is [04](04-event-features.md).

## Common schema

Each inventory produces one table, one row per event, over the same epochs and the same
speed-eligible population (overview X4). Every table carries:

- **Identity** — `animal`, `day`, `epoch`, `brain_area`, `detector`, `event_index`,
  and a detector-scoped key. Keys are never shared across inventories (overview X8).
- **Timing (seconds)** — `event_start_s`, `event_end_s`, `interval_span_s`,
  `event_midpoint_s`. `interval_span_s` is defined identically everywhere as
  `event_end_s - event_start_s`, because the native duration conventions differ: our
  canonical cohort's duration includes a sample-width convention that the sequence
  detector's first-to-last-midpoint duration does not. Native duration fields are
  retained alongside it, never substituted for it.
- **Speed** — `max_speed`, `mean_speed`, `whole_event_speed_eligible`.
- **Provenance** — `run_id` joining to the frozen configuration and artifact manifest.

Detector-specific statistics stay in detector-specific columns, namespaced by detector,
and are never compared across detectors as if they meant the same thing.

**No inventory is a truth set.** The word "detected" is always qualified by which detector.

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
Each event contributes once to its own denominator regardless of how many matches it
has. Show overlap-length and IoU distributions so a one-sample edge contact is not
counted as agreement with a fully coincident event.

**Fragment-merge recount.** Our detector can split one replay into several events when
the non-local probability dips below threshold. Provide a recount in which our events
separated by less than a stated gap (a fixed grid: 0, 10, 20, 50 ms) are merged before
overlap is computed, alongside the native count. The native count remains primary; the
recount is what 04 uses to test whether count differences are segmentation (P11).

**Also report time coverage** of the intersection and union of each pair of cohorts.
Coverage is less sensitive to segmentation than event counts, and the two can disagree —
one detector splitting an event into three does not change the time it accounts for.

**Overlap is not transitive.** A can overlap B and B overlap C without A overlapping C.
Membership in "all six" is therefore not well-defined by pairwise overlap alone.
Summarize connected components of the multipartite overlap graph, labelled by which
detectors contribute to each component, and report component size and span as
diagnostics — a long chain can fuse events that do not individually overlap.
**Do not turn components into replacement event intervals.**

### The cells that pairwise counting misses

The cells of most interest are the ones no pairwise table reaches:

- **Events found by exactly one detector**, per detector.
- **SWRs matched by no sequence detector** — ripples with no decodable replay content.
  This is the reciprocal of the ripple-less-replay claim and the first thing a referee
  asks. It is only computable because SWR inventories are first-class here rather than
  labels attached to someone else's events.
- **Sequences matched by no LFP- or rate-triggered inventory** — events the standard
  field pipelines would not have found at all. This is the neither-ripple-nor-burst
  cell, distinct from ripple-less (overview): an MUA-matched, SWR-unmatched sequence is
  ripple-less but not in this cell.
- **Time accounted for by no detector**, as the denominator that makes the rest legible.

## The SWR label matrix

Distinct from the inventory comparison above, and the reason level 3 needs no separate
sweep (overview X6). Each *sequence* cohort — ours and the memoryless one — is labelled
under each of the three SWR definitions, giving a cohort × SWR-definition matrix of
non-SWR fractions.

A non-SWR fraction is a function of the SWR definition as much as of the events, so it is
never reported without its definition attached. Where the three definitions disagree
about a given event, that disagreement is itself reported: an event labelled ripple-less
by one algorithm and ripple-bearing by another is a finding about detector sensitivity,
not noise to be averaged away.

Keep a separate table of paired label agreement. A matched pair's two events can carry
different SWR labels purely because their bounds differ.

## Secondary analyses

Both are clearly labelled as secondary and never substituted for the primary cohorts.

1. **Native speed gates** (overview X4). Re-run each method under its own published
   gate — Widloski's per-bin < 5 cm/s, ours whole-event < 4 cm/s, Yu's < 4 cm/s — and
   report how much of each cohort difference the gate accounts for.
2. **Running-only encoding** (overview X5). Refit the clusterless encoding model on
   running periods only and re-run **both** decoding-based methods on it. Applied to both
   or neither; applying it to one would make the encoding model a free variable
   distinguishing them.

## Reporting

All main summaries show per-session values and animal-level consistency, with sessions
nested within animal; pooled event summaries are descriptive. Report native duration
distributions before any duration-stratified comparison, since longer intervals have more
opportunity to overlap anything. Report missing features and skipped epochs explicitly —
model and data availability define the paired analysis population.

## Testing

| test | what it protects |
|---|---|
| One-to-many and many-to-many overlaps count each event once per detector denominator | The denominators, which are the easiest thing to get silently wrong |
| Matched events may carry different SWR labels; the paired-label table records it rather than resolving it | Boundary-driven label disagreement stays visible |
| Disjoint cohorts, empty cohorts, and single-event cohorts | Degenerate inputs |
| Connected components: a chain A–B–C where A and C do not overlap is reported as one component with its span, and is not collapsed into an interval | Non-transitivity |
| Coverage and count summaries disagree as expected on a constructed split/merge case | That both are reported because they answer different questions |
| Fragment-merge recount at gap 0 reproduces the native counts; at a gap larger than every inter-event interval it yields one event per contiguous run | The recount is anchored at both ends |
| `interval_span_s` is computed identically for every inventory, and native duration fields are never substituted for it | The convention mismatch between cohorts |
| An event at an epoch boundary is not matched across epochs | Session scoping |
| Empty inventories carry the schema's dtypes, not object, so concatenation across epochs preserves column types | The canonical builder's empty return is all-object |

## Deliverables

1. The six inventories on the common schema, with per-detector counts and rates.
2. The all-pairs overlap table and the per-detector denominators.
3. The single-detector and no-detector cells, including SWRs with no sequence and
   sequences no triggered pipeline found.
4. The cohort × SWR-definition non-SWR matrix, with paired label disagreement.
5. The two secondary analyses.
6. Reproducibly sampled examples from each cell, on a shared time axis showing spikes,
   both posterior representations, gate masks, speed, and LFP — with the sampling rule
   stated and counterexamples included.
