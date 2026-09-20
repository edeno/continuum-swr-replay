# What differs about the events, and why each detector selected them

*Design spec. Status: **deferred** — scoped, not designed in detail. Do not begin until
[03](03-inventory-comparison.md) has produced matched cohorts.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

[03](03-inventory-comparison.md) establishes *which* events each detector found and where
they coincide. This spec covers what the events are like and why a given detector
rejected what another accepted. It is recorded now so the material is not lost, and
deferred because it is not executable until matched cohorts exist.

This is also where **level 2 — scoring** belongs (overview): holding the candidate set
fixed and varying the trajectory score (MAP, weighted correlation, Radon transform,
linear regression, spread/jump/dispersion) to ask how much of "replay" is a scoring
convention rather than a property of the data.

## Event features to report

Duration and span, spike count, active tetrode count, spike rate, spatial extent,
distance from the animal, continuous/fragmented state occupancy, posterior uncertainty
and multimodality, trajectory reversals and pauses, ripple power, and task context (well,
time in stop, reward and patch context where available).

Compute summaries for every detector's intervals from the **same underlying traces**, and
namespace anything that depends on a particular posterior (`ss_*`, `ms_*`). Differences
in posterior confidence or smoothness partly follow from the procedures that generated
them and are not independent evidence about the events.

Reuse existing trajectory and spike-evidence helpers where their timing and definitions
apply, rather than writing parallel implementations that would drift.

## Selection diagnostics

**For events only our state-space detector found**, retain which comparator stage
rejected them: window gates failing, no candidate assembled, or an assembled candidate
failing duration, dispersion, or whole-event speed. Retain *every* failing criterion, not
a single assumed cause, and relate each event to nearby rejected candidates as well as
accepted ones.

Gate-failure summaries explain selection; they do **not** independently validate the
extra events. Any claimed effect of a particular gate must be confirmed by re-running
with only that gate changed.

**For events only the comparator found**, examine the state-space probability trace
together with the Local and No-Spike alternatives over the same interval, plus
prior-excluded spatial evidence and any shorter constituent events. Distinguish a
boundary or segmentation difference from a sustained disagreement about what is
represented. Treat credible missed sequences as limitations of our model, not as errors
of the comparator by definition.

## Controlled changes

Vary one operation at a time: window width and stride, gap merging, spread and jump
gates, duration and dispersion cutoffs, our non-local probability cutoff, and the fitted
temporal priors. **Freeze the sensitivity configurations before inspecting cohort
differences**, retain each run's configuration, and report every setting tested rather
than the ones that worked.

The primary comparison changes several algorithmic choices at once and cannot by itself
isolate the contribution of the temporal prior; that is what these controls are for.

## Reviewer questions this must answer

| Concern | Analysis | What would support the claimed advantage |
|---|---|---|
| Are the extra events just shorter or less spatially extended by definition? | Separate duration, dispersion, and continuity failures; report native cohorts and a comparison within common duration/extent ranges | Additional evidence-supported events within comparable ranges, or a clearly stated expansion to a different representational target |
| Does the state-space prior manufacture non-local events? | Prior-excluded spike-evidence diagnostics; transition-prior sensitivity; held-out evidence or simulation where available | Remote evidence persists without selecting and scoring on the same posterior, and conclusions survive reasonable prior changes |
| Does the sliding window manufacture smoothness, or fuse distinct events? | Window/stride and merge-rule sensitivity; boundary and split/merge analysis | Observed changes follow the predicted temporal operation; simulation establishes which boundaries are recovered correctly |
| Are comparator-only events real sequences our model rejects? | Inspect all four state probabilities and spike evidence on those intervals, including well-supported counterexamples | Rejection is explained without assuming our labels are truth; failures of our Local model or temporal prior are documented where found |
| Is the apparent benefit driven by encoding or threshold choices? | Hold encoding fixed in the primary comparison; vary detector thresholds, and separately fit both procedures on the same running-only training mask | The relevant differences persist under these controls; if not, the claim narrows accordingly |
| Does a detector difference explain the non-SWR finding? | Identical qualified SWRs; duration-stratified non-SWR fractions; boundary-related label disagreements | Non-SWR conclusions persist beyond differences in interval duration and boundary placement |
