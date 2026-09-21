# Deferred validation: what would license a performance claim

*Design spec. Status: **parked**. Recorded so the reasoning is not lost; not scheduled.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

## The distinction this spec exists to preserve

[03](03-inventory-comparison.md) and [04](04-event-features.md) produce a **descriptive**
result: which events each detector found, how they differ, which predictions about each
method held, and which algorithmic step accounts for a difference. That is the first
level of evidence and it is what the report delivers.

Claims about **sensitivity, specificity, or calibration** are a second level and require
one of the two studies below. Without them the interpretation stays descriptive, and the
report says so in those words. More events, smoother trajectories, or larger posteriors
do not by themselves establish that a detector is better. Prediction P14 in 04 — that
our within-event statistics are inflated by scoring content on the posterior that
selected it — is the one row the descriptive analyses cannot reach, and it is the
specific reason either study would be reopened.

## Why parked

Both studies are expensive, and both have design problems that were worked through and
are recorded here so they do not have to be rediscovered. Neither is needed for the
descriptive report.

**Trigger for reopening:** a referee request for a sensitivity or specificity number, or a
decision to make P14 a tested prediction rather than a stated limitation.

## Study 1 — Simulation with planted activity

Plant known activity and score recovery. Generators must include travelling sequences,
stationary remote representation, fragmented activity, local representation, and
no-sequence activity, varying spike counts and event durations, and **must include
generators that do not reproduce our transition assumptions** — otherwise the study is
rigged toward the state-space model.

Score non-local representation and extended travelling sequences as separate targets, so
a detector is not penalized for omitting events outside its remit. A posterior-threshold
detector and a sequence detector are not aiming at the same thing; one target would
mistake a difference in aim for a difference in quality.

This is the only design that can test the failure-mode predictions in 04 (P9, P12, P13)
on activity that is known not to be replay.

## Study 2 — Held-out tetrodes on real data

Split the CA1 tetrodes into a discovery set D and a held-out set H. Run every pipeline on
D alone — encoding restricted to D by subsetting the per-tetrode KDE lists and re-summing
the ground-process intensity, transition parameters re-estimated on D, MUA recomputed on
D, SWR left on LFP — and score each event's D-decoded trajectory on H's spikes.

Two constraints make or break it:

1. **Every pipeline is re-run on D.** Scoring held-out channels after using all channels
   to select events does not qualify: the events were chosen using the data now being
   called held-out.
2. **The scoring rule is frozen before the study runs**, not chosen after seeing which
   detector it favours.

### Design issues worked through

- **Discovery loses sensitivity.** With a quarter held out every pipeline runs on 75 %
  of the spikes; the events found are not the events the paper reports, and transferring
  held-out precision to the full pipelines is an assumption. Fair across pipelines on the
  same folds; understates all absolute numbers.
- **Held-out coverage is patchy.** Five tetrodes do not tile a nine-arm track. Where H has
  no place-tuned marks on the arm D decoded, H's likelihood is flat and the event scores
  as unsupported regardless. Requires an H-coverage map per fold (summed ground-process
  intensity per arm) and support rates stratified by coverage; the covered-arm rate is
  the citable one.
- **No-spike bins are not neutral.** With zero H spikes the clusterless log-likelihood is
  minus the summed ground-process intensity, favouring wherever H fires least. A
  trajectory into a low-rate arm is "supported" by silence. A shift null absorbs most of
  it, but it is a systematic term.
- **State-space decoders tolerate sparsity better than memoryless ones** because the
  prior fills gaps — either a genuine advantage or the prior compensating for missing
  evidence, and the split alone cannot say which.
- **Fold count is a measurable trade-off.** K = 4 (H ≈ 25 %) versus K = 8 (H ≈ 12 %):
  pilot on three sessions, recording per pipeline the fraction of full-data events
  recovered on D and the width of the shift null on H, before freezing K.
- **Session floor.** A stated minimum of CA1 tetrodes with spikes (about 12), decided in
  advance; sessions below it are excluded, not scored badly.

### Score

Four candidates were considered, all computed from H's spikes and H's encoding against
D's trajectory:

1. Likelihood at the decoded position minus at the animal's (the existing `dll_animal`):
   targets non-local content directly, but is zero for local content, unbounded in spike
   count, and carries the no-spike bias.
2. Held-out log-likelihood along the trajectory versus a circular-shift null: drops the
   animal reference so local content can score; still spike-count scaled.
3. **Held-out posterior probability of the discovery trajectory relative to uniform** —
   decode H alone with a memoryless uniform-prior decoder, then take the mean over bins
   of `log P_H(x̂_D(t)) − log(1/n_bins)`. Bounded, per-bin normalized, so a bin with no H
   spikes contributes near zero rather than an artefact, and no bin can dominate.
4. Per-spike version of 1: fixes scale only.

If reopened, freeze 3 as primary with a circular-shift null (same event, same H spikes,
same D trajectory, shifted over the session's immobility time), and report each pipeline
over all its events and over the subset whose D-trajectory is non-local, so the SWR
pipelines are not penalized for local content the score was never meant to reward.

### Alternatives considered

| design | independence tested | limitation |
|---|---|---|
| Tetrode split (above) | generalization across neurons | coverage, sparsity |
| Spike thinning (random half of each tetrode's spikes to D or H) | Poisson consistency only — same neurons on both sides | a single bursting cell shows up on both sides; cannot show a population phenomenon |
| Trigger on all tetrodes, re-decode trajectory on D, score on H | trajectory reality only | selection leaked; violates constraint 1 |

Spike thinning is worth running as a diagnostic control alongside the tetrode split:
where thinning supports an event and the split does not, the gap is coverage or
single-cell dependence, not a failed decode.

## Relationship to the report

The report does not wait for either study. It states its interpretation as descriptive
and says so explicitly. If a study is later run and supports a performance claim, that
claim is made then — not retrofitted onto the descriptive result.
