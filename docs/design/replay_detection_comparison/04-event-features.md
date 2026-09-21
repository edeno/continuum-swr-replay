# Predictions and diagnostics: how the events differ, and why

*Design spec. Status: approved, not yet implemented. Do not begin the analyses until
[03](03-inventory-comparison.md) has produced matched cohorts; do not edit the
predictions table after it has.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

[03](03-inventory-comparison.md) establishes *which* events each detector found and where
they coincide. This spec covers what the events are like, why a given detector accepted
or rejected them, and what that says about each method's strengths and weaknesses —
ours included. It has no ground truth and claims none; what it offers instead is a set of
predictions written before the results exist, each with the observation that would count
against it.

## Principle

The comparison is objective to the extent that its predictions precede its results.

- The predictions table below is **frozen** (overview X9): committed, hash recorded in
  the run manifest, before 03 produces its first matched cohort. No row is added,
  removed, or reworded after that point. A prediction that turns out to be badly posed
  is reported as badly posed, not replaced.
- **Every row is reported**, whichever way it comes out. Rows that come out against our
  state-space detector become stated limitations in the methods and discussion, with the
  same prominence as rows that come out against the other methods.
- A prediction that is confirmed for a method is a *characterization* of that method,
  not a demonstration that another method is better. Whether a difference is a flaw or
  a different scientific target is argued per row, in the text, from the evidence.

## The predictions table

Each row names the method, the predicted weakness, the analysis that tests it (defined in
the next section), and the result that would count against the prediction. The last
column is always evidence **against the prediction**, so for the rows about our own
detector it is the outcome that would *favour* our detector; read it that way.

| # | method | predicted weakness | tested by | evidence against |
|---|---|---|---|---|
| **P1** | SWR (Kay, Karlsson, Yu) | Ripple power is continuous, not bimodal, so the threshold discards a lower-power tail whose events carry real spatial content | A2 threshold sweep; A1 ripple-power feature on ours-only events | Ours-only events have ripple power indistinguishable from non-event immobility |
| **P2** | SWR | Admits events whose spatial content is local or absent | A1 state-occupancy and distance features on SWR-only events; A3 spike evidence | SWR-only events carry non-local spike evidence at the rate shared events do |
| **P3** | SWR | Boundaries follow the LFP envelope, not the content | A4 onset/offset differences on matched pairs | Offsets are small and symmetric |
| **P4** | SWR | The three SWR algorithms disagree with each other about the lower-power tail | 03's SWR-label matrix; A2 sweep per algorithm | The algorithms' unique cells are empty or tiny |
| **P5** | MUA | As P1 and P2, and the event set depends on which tetrodes are present | A1, A2, A3 as for SWR; A5 leave-one-tetrode-out stability | Stable event sets under tetrode removal |
| **P6** | Widloski | The 80 ms window blurs onsets and merges close events | A4 boundary differences and split/merge counts on matched pairs | No systematic widening; one-to-many counts match ours |
| **P7** | Widloski | The 100 ms duration and 12 cm dispersion criteria reject short and stationary remote representation that our detector counts | A1 duration and dispersion features on ours-only events | Ours-only events are mostly above both cutoffs |
| **P8** | Widloski | A flat memoryless posterior under sparse spiking fails the spread gate, so low-spike events are lost | A1 spike-count and tetrode-engagement features, ours-only vs Widloski-only; A6 gate-failure diagnostics | No spike-count difference; gate failures are not dominated by spread |
| **P9** | **Ours** | The temporal prior manufactures non-local states when spikes are sparse | A3 prior-excluded spike evidence on ours-only events, against the matched immobility baseline | Ours-only events show likelihood support for the decoded position above the baseline, at a rate comparable to shared events |
| **P10** | **Ours** | A position-tracking error turns local activity into "non-local" | A1 distance-from-animal on ours-only events; relation to tracking confidence where available | Ours-only events are distributed across distances like shared events and show no association with low tracking confidence |
| **P11** | **Ours** | The 0.8 probability threshold and 15 ms minimum fragment one replay into several events, inflating counts | A4 one-to-many overlaps; 03's fragment-merge recount | The count excess over SWR and Widloski persists after fragments within a stated gap are merged, i.e. it is not segmentation |
| **P12** | **Ours** | Acausal smoothing shifts event boundaries | A4 onset relative to ripple onset and MUA onset on matched pairs | Offsets near zero, no sign bias |
| **P13** | **Ours** | The Fragmented state admits noise | A1 state composition of ours-only events; A3 spike evidence for fragmented-dominated events | Fragmented-dominated events carry support comparable to continuous ones |
| **P14** | **Ours** | Content is scored on the posterior that selected it, so selection inflates every within-event statistic | Not testable by the analyses here; see [05](05-validation.md) | — |
| **P15** | All | Removing one tetrode changes the event set more for spike-triggered and posterior-triggered methods than for LFP-triggered ones | A5 | Comparable stability across families |

P14 is listed so that its absence from the results is deliberate rather than an oversight.

## Analyses

Six analyses. Each is computed for **every** inventory from the **same underlying
traces**, so a feature never differs across inventories because of how it was computed.
Anything derived from a particular posterior is namespaced (`ss_*` for the state-space
posterior, `ms_*` for the memoryless one, `cf_*` for ContFrag). Existing trajectory and
spike-evidence helpers are reused where their timing conventions apply.

### A1 — Feature contrasts on the cells

For each cell of the overlap accounting in 03 — shared, unique-to-one, and the
neither-ripple-nor-burst cell — report the distribution of:

- **Timing:** `interval_span_s`, native duration, gap to the nearest event of each other
  inventory.
- **Population engagement:** spike count, active tetrode count, spike rate, all from the
  raw marks over the interval.
- **LFP:** ripple consensus power (Kay trace, z-scored to immobility) and its peak;
  reported for events of every inventory, including the SWR ones, so that "ripple power
  of an ours-only event" and "ripple power of an SWR event" are on one axis.
- **MUA:** population rate z-score, same convention.
- **Spatial content:** distance of the decoded position from the animal, spatial extent,
  trajectory reversals and pauses, posterior spread; namespaced per posterior.
- **State composition:** `ss_*` occupancy of Local, No-Spike, Continuous, Fragmented over
  the interval; reported for every inventory's events, which is what tests P2 and P13.
- **Task context:** well, time in stop, reward and patch context where the stop table
  supplies them.

Feature distributions are shown per cell with per-session values and animal-level
consistency, never pooled only.

### A2 — Threshold sweeps

For each method, sweep its own defining threshold over a stated grid and re-run only
that method:

| method | swept parameter | grid |
|---|---|---|
| Kay, Karlsson | z-score threshold | 1.0–4.0 in 0.25 |
| Yu | percentile | 99, 99.9, 99.99, 99.999 |
| MUA | z-score threshold | 1.0–4.0 in 0.25 |
| Widloski | spread, jump, duration, dispersion — one at a time | ±50 % of the retained value in 5 steps |
| Ours | non-local probability; minimum duration — one at a time | 0.5–0.95 in 0.05; 5–30 ms in 5 |

For each setting report two things against each other inventory at *its* retained
setting: the fraction of that inventory's events now captured, and the total event time
now admitted. The yardstick is therefore model-free — time, not any model's labels. The
spike evidence (A3) of the events admitted at each relaxation is reported alongside, so a
reader can see whether a lower SWR threshold captures ours-only events or merely admits
more time. The sweep is what tests P1, P4, and P5 directly.

### A3 — Prior-excluded spike evidence

`scripts/qa/analyze_event_spike_evidence.py` already computes, for our events, the
clusterless spike log-likelihood over track bins with the temporal prior removed, and
scores whether it favours the decoded position over the animal's own, against a matched
non-event immobility baseline. Extend it to every inventory's events and to every cell of
03. For events without a decoded position (SWR, MUA), score the maximum over remote bins
(beyond the QA script's `REMOTE_CM` track distance from the animal) against the
animal's position, and report that this is a weaker statistic.

This analysis tests P2, P9, and P13. Its caveat is P14 and is stated with every use: the
same spikes selected the event, so support is necessary, not sufficient. The baseline
comparison, not the raw statistic, is the reported quantity.

### A4 — Boundaries and segmentation

On the matched-pair table from 03: signed onset and offset differences per pair of
inventories, one-to-many and many-to-one counts, and the fragment-merge recount (merge
our events separated by less than a stated gap, then recount overlaps). Onsets are also
referred to two independent markers, ripple onset and MUA onset, for every inventory,
including the ones that were not triggered by that marker. Tests P3, P6, P11, P12.

### A5 — Robustness to recording

Leave one tetrode out and recompute the event set; report the overlap (interval
Jaccard and event-count change) between the full and reduced event sets, per tetrode,
per method. For the triggered and memoryless methods this needs no full re-decode: MUA
recomputes the population rate, the SWR consensus drops a channel, the memoryless
likelihood drops a tetrode's terms. For our state-space detector it needs a re-decode per
tetrode, so it runs on a stated subset — the two sessions per animal with the most CA1
tetrodes carrying spikes, chosen before the analysis — and that restriction is reported. Tests P5 and P15.

### A6 — Selection diagnostics

For events only our detector found, retain which comparator stage rejected them: window
gates failing (and which), no candidate assembled, or an assembled candidate failing
duration, dispersion, or whole-event speed. Retain *every* failing criterion, not a
single assumed cause. For events only the comparator found, examine the state-space
probability trace with the Local and No-Spike alternatives over the same interval, plus
the prior-excluded evidence, and distinguish a segmentation difference from a sustained
disagreement about what is represented. Credible missed sequences are limitations of our
model, not errors of the comparator by definition. Gate-failure summaries explain
selection; any claimed effect of a particular gate is confirmed by re-running with only
that gate changed (A2). Tests P7 and P8.

### Task validity (secondary)

Re-run the primary rate and content contrasts with each inventory in turn as the event
definition, using the existing ladders unchanged. Report whether the reward and patch
effects appear under every definition. This is reported as **robustness**: an effect
that is stronger under our events could be either sensitivity or selection, and this
analysis cannot tell them apart, so it is never cited as superiority.

## Level 2 — scoring (deferred)

Holding the candidate set fixed and varying the trajectory score (MAP, weighted
correlation, Radon transform, linear regression, spread/jump/dispersion) asks how much of
"replay" is a scoring convention rather than a property of the data. It belongs here and
is deferred until the level 1 comparison above is complete.

## Reporting

All summaries show per-session values and animal-level consistency, sessions nested
within animal. Each prediction row gets one figure or table that answers it and one
sentence of verdict in the results, in the row order above. Every sweep setting tested is
reported, not the ones that worked. Missing features and skipped sessions are reported
explicitly.

## Testing

| test | what it protects |
|---|---|
| Every feature is computed from the same traces for all inventories; two inventories with an identical interval get identical feature values | The contrasts measure the events, not the pipelines |
| Namespaced posterior features cannot be joined across posteriors by name | No silent `ss_`/`ms_` mix |
| Threshold sweep at the retained setting reproduces the primary inventory exactly | The sweep is anchored |
| Fragment-merge recount at gap 0 reproduces the unmerged counts | The recount is anchored |
| Leave-one-tetrode-out with a tetrode that has no spikes reproduces the full event set | A5 measures the tetrode, not the recompute |
| Selection diagnostics agree with a gate-specific re-run on synthetic examples | Explanations follow the actual pipeline |
| The predictions table in the committed spec matches the hash in the run manifest | X9 |

## Deliverables

1. The frozen predictions table with its commit hash.
2. Feature-contrast figures per cell (A1).
3. Sweep curves per method with spike evidence of admitted events (A2).
4. Spike-evidence distributions per inventory and per cell (A3).
5. Boundary and segmentation summaries, including the fragment-merge recount (A4).
6. Tetrode-stability table (A5), with the session subset stated.
7. Selection-diagnostic tables for both unique cells (A6).
8. The task-validity robustness table.
9. A verdict per prediction row, in order.
