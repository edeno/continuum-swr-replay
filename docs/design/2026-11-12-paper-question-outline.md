# Proposed organization of the paper's questions

**Date:** 2026-11-12. **Status:** proposed outline, recorded at the project lead's
request. It develops the [paper-questions review](../../.claude/docs/reviews/2026-11-11-paper-questions-review.md)
and accompanies the [paper-development plan](2026-11-11-paper-development-plan.md).
The [question registry](../../src/continuum_swr_replay/reported/questions.py)
remains the current maintained inventory; the mappings below refer to its existing
identifiers and propose how to group them.

## Terminology

**Replay includes stationary representations as well as trajectories.** This was
explicitly clarified by the project lead on 2026-11-12 and is recorded in the
[development guide](../development.md#replay-terminology).
Stationary describes the represented location remaining stable; it does not refer
to whether the animal is moving. Use "trajectory replay" or "sequential replay"
when a question or comparator specifically requires movement of the represented
location. Such a requirement does not define replay as a whole.

Validation should establish support for represented spatial content in both
stationary and trajectory events. Dynamics characterize that content; a stationary
representation is not a failed trajectory or a non-replay control simply because
it does not move. Decoder-state labels and their validation remain a separate
measurement question.

## Overarching question

> What non-local hippocampal representations occur outside detected SWRs, and how
> are they associated with behavior and cortical activity?

The introduction establishes why identifying these representations matters, what
existing detection methods measure, and what remains unknown. The literature
survey belongs there. The main questions below organize the Results; supporting
measurements and controls sit beneath the question they help answer.

## 1. Can we reliably identify non-local representations from hippocampal spiking?

1. **Detection:** When the represented location is known, can the model distinguish
   local activity, non-local spatial content, and activity without identifiable
   spatial content, including recovery of stationary representations and trajectories?
2. **Independent validation:** Do spikes excluded from detection support the
   inferred representation?
3. **Scope:** Which event populations are supported—Continuous, Fragmented, close,
   and remote—and where does the method fail? State how these decoder-defined
   populations relate to stationary and moving representations.

**Supporting analyses:** simulated recovery and false detections, held-out decoding,
matched windows, Local-kernel sensitivity, prior sensitivity, and reduced-model
comparisons. A spatially supported stationary event is a positive replay target;
place-less activity is a different control.

**Current questions:** 1.1, 2a.1–2, 2a.4, and 2c.2–7.

## 2. How does non-local representation relate to SWRs and other event definitions?

1. **Occurrence:** How much independently supported non-local activity occurs
   inside and outside detected SWRs?
2. **Timing:** How do the boundaries of non-local representations relate to SWR
   boundaries?
3. **Detection dependence:** Which discoveries are shared or unique across
   detectors, and how sensitive are these conclusions to the detection settings?

**Supporting analyses:** overlap inventories, continuous ripple-power distributions,
threshold sweeps, electrode-level checks, and behavioral-state characterization.
Examples illustrate the answers rather than constituting a separate question.
When comparing with a trajectory detector, distinguish its narrower target from
performance on targets both methods attempt to detect. A stationary event missed
by that detector is not automatically a failure of its implementation.

**Current questions:** 2a.3, 2a.5, 2b.1–5, and 2c.1.

## 3. What do these events represent, and how do their representations evolve?

1. **Location:** Where is represented content relative to the animal, reward wells,
   and other parts of the environment?
2. **Dynamics:** Does content remain stationary, move along a trajectory, or change
   discontinuously? How do these patterns occur within and across events?
3. **Relationship to SWRs:** How do these properties vary with SWR overlap and
   ripple strength?

**Supporting measurements:** distance, duration, extent, speed, pauses, reversals,
and posterior uncertainty. These measure content and dynamics rather than defining
separate headline questions. Duration and decoding quality inform how comparisons
should be interpreted. Stationary content and trajectories are both within the
paper's replay scope.

**Current questions:** 2c.8 and 2d.1–6.

## 4. How are event occurrence and content associated with behavioral history?

1. **Occurrence:** How does recent experience, particularly reward history, relate
   to when events occur?
2. **Content selection:** Which locations are represented more than expected after
   accounting for spatial and behavioral baselines? What additional associations
   remain with reward history, inferred value, or uncertainty?
3. **Subsequent choice:** Does event content provide information about the next
   choice beyond an appropriate behavioral baseline?

Keep these questions distinct: when an event happens, what it represents, and
whether that representation adds information about subsequent behavior. Compare
SWR and non-SWR events within each question. Stationary replay can contribute
location content without representing a path to a choice.

**Supporting analyses:** visit recency, occupancy, stop timing, task structure,
alternative content estimators, and held-out prediction where prediction is
claimed. Negative choice results remain part of the answer.

**Current questions:** reorganize 3.1–7; use 3.8 to support the relationship between
behavioral context and ripple coupling where it advances a main conclusion.

## 5. How is mPFC activity associated with non-local hippocampal events?

1. **Event association:** Is mPFC activity systematically related to events outside
   detected SWRs?
2. **Specificity:** Does that association remain after accounting for shared
   behavioral context, hippocampal population activity, and ripple power?
3. **Content dependence:** Does the mPFC response depend on what the hippocampus
   represents?

Compare SWR-associated and non-SWR events within these questions. Content dependence
is a stronger extension with its own evidence requirements; an event-associated
response alone does not answer it. Content dependence can concern represented
locations, including stationary representations, without requiring trajectory
information.

**Current questions:** split 4.1 into these distinct scientific questions.

## Organization within each question

Use **primary comparison → essential controls → secondary characterization**.
Show animal-level consistency and uncertainty throughout rather than making them
a separate scientific section. Keep computed status separate from whether the
scientific question has been adequately answered.

The main questions need not have equal space or one figure each. Questions 2 and 3
can share a figure; detailed characterization can move to the supplement. Keep
the main text focused on evidence that changes the interpretation, while preserving
the full analysis inventory beneath the outline.
