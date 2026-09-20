# Validation: what would license a claim about detector performance

*Design spec. Status: **deferred** — scoped, not designed in detail.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

## The distinction this spec exists to preserve

Specs [03](03-inventory-comparison.md) and [04](04-event-features.md) produce a
**descriptive** result: which events each detector found, how they differ, and which
algorithmic step accounts for a difference. That is the first level of evidence and it is
what the initial report delivers.

Claims about better **sensitivity, specificity, or calibration** are a second level and
require this study. Without it, the comparison's interpretation stays descriptive.

This is recorded as its own spec rather than as a caveat paragraph precisely because
caveats get dropped and structure does not. More events, smoother trajectories, or larger
posteriors do not by themselves establish that a detector is better.

## Simulation study

Plant known activity and score recovery. Generators must include:

- travelling sequences,
- stationary remote representation,
- fragmented activity,
- local representation,
- no-sequence activity,

varying spike counts and event durations, and **including generators that do not
reproduce our transition assumptions** — otherwise the study is rigged toward the
state-space model.

Score **non-local representation** and **extended travelling sequences** as separate
targets, so a detector is not penalized for omitting events outside its remit. A
posterior-threshold detector and a sequence detector are not aiming at the same thing;
scoring them against one target would mistake a difference in aim for a difference in
quality.

## Real-data validation: held-out tetrodes

Detect on a discovery set of tetrodes, then score neural evidence on held-out tetrodes,
keeping encoding and threshold tuning entirely separate from evaluation.

Two constraints make or break this:

1. **Both detectors must be re-run on the discovery set.** Scoring held-out channels
   after having used all channels to select events does not qualify — the events were
   already chosen using the data now being called held-out.
2. **The predictive scoring rule must be defined before the study is run**, not chosen
   after seeing which detector it favours.

Prior-excluded likelihood evaluated on the *discovery* spikes remains a useful
diagnostic, but it is not independent validation: it scores the events on the same data
that selected them.

## Relationship to the initial report

The initial report does not wait for this study. It states its interpretation as
descriptive and says so explicitly. If this study is later run and supports a performance
claim, that claim is made then — not retrofitted onto the descriptive result.
