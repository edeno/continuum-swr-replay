# Timing, false positives, multi-part events and held-out evidence in the memoryless comparison

Design for extending the simulated comparison of the memoryless sliding-window
detector (spec 02) and our state-space detector. Agreed 2026-09-25.

## Why

The injected-replay sweep measures only sensitivity: every cell holds exactly one
straight, constant-speed replay. Three things a reader needs are missing:

1. **Timing.** The detectors differ most in where an event starts and ends, but the
   figure shows only the reported/actual duration ratio, which hides whether extra
   length comes before or after the event.
2. **Specificity.** Nothing is injected that is *not* replay, so false positives are
   unmeasured, and more detection cannot be read as better performance.
3. **Events that are not one clean trajectory.** Replay can jump mid-event, and replays
   come in chains (for example across a ripple train). How each detector segments such
   activity is a difference between the methods that the sweep cannot show.
4. **Independent support for the additional discoveries.** Our detector reports many
   events the sliding window does not. The compelling result would be that those
   additional detections carry reproducible spatial information in neurons excluded
   from detection and tuning. The simulation is where that held-out test can be
   calibrated, because there the truth of every event is known (spec 05's held-out
   study, tried here first).

## Decisions

| # | Decision | Reasoning |
|---|---|---|
| C1 | Our detector is scored two ways throughout: canonical (Non-Local Continuous + Fragmented > 0.8 for >= 15 ms, the project's event definition) and continuous-only (same thresholds on Non-Local Continuous alone) | An incoherent burst decoded as fragmented is a correct non-local call under our model but not replay; reporting both separates the two |
| C2 | Two kinds of burst: a place-less burst and a jumping burst | They test different criteria: the place-less burst gives a broad posterior (spread gate), the jumping burst sharp posteriors that hop (jump gate, and our fragmented state) |
| C2a | Truth has two targets, scored separately: *non-local representation* and *continuous replay*. A continuous replay is both; a jumping burst is non-local representation but not continuous replay; local coding and a place-less burst are neither | A jumping burst represents real places every 15 ms, so calling it non-local, or finding held-out spatial information in it, is correct; only calling it a sequence is an error. A single "negative" label would score correct answers as false |
| C3 | Multi-part events are scored per part and by segmentation, with no outcome declared correct | Whether a jump or a chain is one event or several is a question of definition; the figure shows how each method carves the same activity |
| C4 | Everything goes in the existing comparison figure, which grows from four rows to six | Keeps the comparison in one place |
| C5 | Today's panel (h), reported/actual duration by true duration, is replaced by signed onset and offset error | The two errors decompose the ratio and show strictly more |
| C6 | Held-out evidence: detect on discovery cells, assess content with evaluation cells excluded from detection and tuning; identical folds and evaluation rule for every method; comparator-only events tested too | Tests whether our additional detections carry reproducible spatial information in independent neurons |
| C7 | Every event, whichever method found it, is evaluated under **both** discovery-derived content estimates: our decoder's non-local posterior p(x \| non-local, discovery spikes) on the native 2 ms grid, and the sliding window's own posterior (80 ms windows, uniform prior, the reference's floor), each 2 ms bin taking the window whose midpoint is nearest. Both come from refits on discovery cells only | Holding out neurons prevents reusing data but does not remove the asymmetry of letting one method's decoder judge every method's content: a correct comparator event our decoder mis-estimates would be under-credited, and calibration in one simulator does not establish fairness on recordings. Using both estimators on every inventory makes the dependence visible |
| C8 | A conclusion about held-out support is stated only when it holds under both content estimators; where they disagree, the disagreement is the result. In simulation each estimator's bias is also measured: true events of either class that fail to get support under it | The truth is known there, so under-crediting by either estimator is measurable |
| C9 | Folds: two interleaved (alternate cells by field position, roles swapped), both covering the track, plus one whose evaluation cells cover only half the track | Shows the score at good coverage and how it misleads at poor coverage |
| C10 | Held-out support is read with the simulation's error estimates, never alone | Held-out support does not reveal missed events or establish real-data sensitivity |

## Simulated inputs

Same simulated recording, immobile placements (labeled events cut out), 8 seeds, and
peak rates (20, 60, 160 Hz) as the injected-replay sweep.

| Condition | What is injected | Grid |
|---|---|---|
| No replay | Nothing: each usable immobile stretch is decoded as it is, holding only local coding | All usable immobile time of eight simulated recordings, the sweep's and seven more (one holds only 12.6 s) |
| Place-less burst | Every cell fires at one rate unrelated to its field; population spike count matched to a replay of the same duration and peak rate | 80, 200, 500 ms |
| Jumping burst | Represented place redrawn uniformly along the track every 15 ms; cells fire by their fields | 80, 200, 500 ms |
| Mid-event jump | Two constant-speed segments of 60 cm; the second starts J cm from where the first ended, with no gap in time | J = 0 (continuous control), 20, 40, 80 cm; segment length 120 or 200 ms |
| Chain | Two 150 ms, 60 cm replays separated by a silent gap; the second either continues from the first's end or starts >= 40 cm away | Gap 10, 30, 60, 100, 200 ms |
| Isolated part | One part of each multi-part geometry on its own: a 60 cm replay of 120, 150 or 200 ms, at the same placements | The three part lengths |

Parts are 60 cm and longer than 100 ms, so each can meet the published definition,
whose duration criterion is strict: a part of exactly 100 ms fails it however well it
is decoded. The isolated parts are the baseline: a part is lost *to the jump or the
chain* only as far as it is recovered less often inside the multi-part event than on
its own.

## Scoring

Five outputs per cell or stretch: the sliding window accepted on duration and
dispersion (before its significance test) and published (after it; the place-cell
identity shuffle runs on every accepted candidate, as in the sweep); our detector
canonical and continuous-only (C1).

**Controls.** No replay: reported events per minute of eligible immobility and the
share of immobile time they cover; an event belongs to the stretch holding its
midpoint. Bursts: share of bursts with at least one reported event overlapping them;
for our detector, also the share of burst time in the continuous and in the fragmented
state. Read against C2a: an event reported on local coding or a place-less burst is a
false positive for either target; an event reported on a jumping burst is correct for
non-local representation and a false positive only as continuous replay, so the
sliding window's and our continuous-only calls on it are the replay false positives,
and our canonical call is not.

**Timing.** Signed onset error (reported start - true start) and offset error (reported
end - true end), in ms, from the event that answers for the cell under the existing
scoring rule. The sliding window's bounds stay its first and last window midpoints; the
caption says its 80 ms windows reach 40 ms past each bound. Summarized as median and
interquartile range by true duration, over replays both detectors reported (paired).

**Multi-part events.**
- Match: the existing whole-event rule against a piecewise true path. A reported event
  matches when its decoded positions sit on the true path at their own times
  throughout, and it does not cover far more ground than the union of the parts.
  Windows in a chain's silent gap have no true position: they are left out of the
  position check but count toward the event's time extent.
- Part recovery: a matching event overlaps at least half of the part's duration.
- Segmentation, one label per trial: one event spanning all parts; one event per part;
  parts missing; other (more pieces, or partial coverage).
- Timing: per part when the detector reports one event per part; against the first
  part's start and last part's end when it reports one spanning event.

## Held-out evidence

**Folds.** The 25 simulated cells, ordered by field position: fold A evaluates the
even-indexed cells and discovers on the odd, fold B the reverse; fold C evaluates the
even-indexed cells in one half of the track only and discovers on all the others. For
each fold, our encoding model is refit on the discovery cells (same non-event training
data and settings), the sliding window decodes that model with the reference's floor,
and the evaluation cells' fields are fitted separately on the same training data. Both
detectors then run on every injected-replay cell and every condition above, from
discovery cells only.

**Event classes.** In each fold, every event is shared (strictly positive overlap with
an event of the other method, the inventory comparison's rule), ours-only, or
comparator-only; and, from the simulation's truth (C2a), continuous replay, non-local
but not a sequence (a jumping burst), local coding (not the injection, or absorbed), or
no spatial content (a place-less burst).

**Evaluation rule, identical for every event.** Over the event's own interval, on the
2 ms grid:

- content: each of the two discovery estimates of C7, scored separately;
- score: the evaluation spikes' expected log-likelihood under that content,
  sum over bins of sum_x p(x) sum over evaluation cells of [k log(f(x) dt) - f(x) dt],
  kept as a spike term and a silence term;
- gain, per evaluation spike, against two nulls: the animal's own position over the same
  bins (does the event beat local coding?), and 100 cell-identity shuffles of the
  evaluation fields (does it beat chance?). The p-value is
  (1 + number of shuffles scoring at least the observed) / 101, and an event has
  held-out support when it is at most 0.05 and its gain over the animal's position is
  positive;
- coverage: expected evaluation spikes under the content over the interval. Below 5 the
  event is uninformative, not unsupported; 2 and 10 are the sensitivity. Fixed before
  any result is seen.

**What the simulation reports.** Per fold, event class and content estimator: the
false-support rate (share of local-coding and place-less events with held-out support),
the support rate for events with spatial content (continuous replays and jumping bursts,
reported separately), and each estimator's under-crediting (events with spatial content
it fails to support, by which method found them), with fold C beside A and B, and the
agreement between the two estimators event by event. Held-out support tests spatial
information, not sequence structure, so it is expected for jumping bursts. These are
the error estimates the real-data held-out results will be read with.

## Code layout

- `memoryless_sequence/simulation.py`: one general injector taking the represented
  position over time (NaN where silent) and a per-cell rate. `inject_replay` keeps its
  signature, is rebuilt on it, and draws its random numbers identically. New builders
  for the place-less burst, the jumping burst, the two-segment jump and the chain.
- `memoryless_sequence/scoring.py`: `score_against_injection` accepts a piecewise true
  path that is NaN in gaps; new pure functions for part recovery and segmentation.
- `scripts/qa/sweep_memoryless_detection.py`: each detector's per-window function is
  split into "run: reported events with bounds and decoded positions" and "score", so
  the new script can reuse the first. `summarize` gains paired onset/offset rows by
  duration. Every existing summary row must be unchanged.
- `scripts/qa/sweep_memoryless_controls.py` (new): runs the conditions above; raw table
  to `Processed-Data/replay_detection_comparison/memoryless_controls.parquet`,
  aggregates and provenance to `detection_controls_summary.csv` beside this file.
  About 580 injected cells plus every no-replay stretch; about a minute at the sweep's
  0.045 s per cell, measured on one seed before the full run. A condition that cannot
  be placed raises; nothing is dropped silently.
- `analysis/inventory_comparison/heldout.py` (new, src, unit-tested): the fold builder
  and the evaluation rule as pure functions of event intervals, spike times, a content
  posterior on the decode grid (either estimator) and the evaluation fields, so the real-data pilot applies the same rule
  unchanged.
- `scripts/qa/sweep_memoryless_heldout.py` (new): the three folds over the injected
  replays and the conditions above; raw table to `Processed-Data`, aggregates to
  `detection_heldout_summary.csv` beside this file. About 3 x 5100 cells at the sweep's
  0.045 s per cell, near 12 minutes, measured on one seed of one fold first.
- `scripts/figures/generate_memoryless_comparison.py`: reads all three summaries.

## Figure

| Row | Panels |
|---|---|
| 1 | (a) overview, unchanged |
| 2 | (b-e) four events, unchanged |
| 3 | (f, g) detection heatmaps, unchanged |
| 4 | (h) onset error by true duration; (i) offset error by true duration; (j) reported duration by distance from the animal (today's panel i) |
| 5 | (k) bursts, with the no-replay result stated on the panel; (l) mid-event jumps; (m) chains continuing; (n) chains elsewhere |
| 6 | (o) held-out gain by event class (shared, ours-only, comparator-only) split by truth, under each content estimator, fold C hatched; (p) false-support and support rates by class, fold and estimator |

(h, i): signed median with interquartile band, one line per detector, zero marked.
(k): no replay as events per minute and the two bursts as share called, grouped by
condition, four bars per group (sliding window before and after the shuffle; ours
canonical and continuous-only). (l, m): stacked segmentation classes per jump size or
gap, one bar per detector, part-recovery rate printed above.

The caption gains paragraphs for (h, i) and (k-m), every number read from the
summaries; each sentence stating a fact gets a check in the generator, so the figure
refuses to build when a claim stops holding. The 40 ms example moves to (h, i) as onset
plus offset.

## Testing

- Unit: each new injector produces the intended path and spike count; `inject_replay`
  output unchanged for a fixed seed; piecewise scoring, part recovery and segmentation
  on hand-built events; mutation-check the new scoring tests.
- Refactor: the sweep's existing summary rows identical after the split.
- Held-out rule on hand-built cases: evaluation spikes drawn from the content's place
  give positive gain; from elsewhere, negative; shuffled fields, near zero; silence
  where the fields predict firing lowers the silence term. Folds disjoint; A and B each
  cover the track.
- Smoke: one seed of the controls script, and one seed of one held-out fold, timed,
  before the full runs.

## Out of scope

Sorted spikes on a linear track only, as the sweep: clusterless decoding, the
encoding-pair shuffle and branched-track jumps are the real-data pilot's (PLAN.md
"Phase 2 pilot"), as is running the held-out rule on recorded tetrodes. No threshold of
either detector, and neither held-out parameter, is tuned against these results.
