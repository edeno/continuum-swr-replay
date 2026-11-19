# Event history when events have duration: the simulation behind the at-risk design

**What this is.** The evidence for two choices of the per-bin rate models (Family B,
[../rate_models.md](../rate_models.md)): a bin inside an ongoing event of the outcome's
detector is not at risk, and history is read from past events' ends as well as their
onsets (`MODEL_CROSS_HISTORY_DURATION_RATE` in `analysis/replay_models/model_specs.py`;
the runner is `scripts/replay_models/rate/fit_duration_aware_history_rate.py`). The evidence
is a simulation, run 2026-10-26; no real data was read. Its numbers describe the
simulation and are not reported values. The dated report with every table is
[2026-10-26-duration-aware-history-simulation.md](../../.claude/docs/reviews/2026-10-26-duration-aware-history-simulation.md).

## The question

Non-local events are intervals, and one decoder's events never overlap. The earlier
per-bin design treated each event as an instant onset: every 15 ms bin of a stop was at
risk, including the bins inside an ongoing event, and self- and cross-history counted
past onsets. On that design the real fits showed a deep negative coefficient at the
shortest cross-history lag and a self-history "burst" about 105–120 ms after an onset.
Are these artifacts of event duration, or real refractoriness, rhythm and coupling?

## How it was checked

`analysis/replay_models/rate/duration_history_simulation.py` simulates two streams of
non-overlapping events (SWR-coupled and non-SWR) with lognormal durations under four
known truths: (a) no dependence beyond non-overlap; (b) fragmentation, a rise in both
streams' rate just after any event's end; (c) an onset-locked rhythm in each stream;
(d) a one-way end-locked coupling from SWR-coupled to non-SWR events. Each simulated
data set is fit by the real design builder and fitter (`aggregate_to_timebins`,
`freeze_per_bin_cohort`, `fit_per_bin_rate_model`), on the onset-only all-bins design
and on the at-risk design with onset and end history, pooled and per simulated session.
The recovery is pinned, seeded, in
`tests/analysis/replay_models/rate/test_duration_history_simulation.py`.

## What it found

- **Non-overlap alone fools the onset-only design.** Under (a) it reports a deep
  short-lag self "refractoriness", a deep, often separated, negative short-lag cross
  coefficient, and a cross-history AIC gain in both directions in almost every session.
  None of these is a dependence in the hazard.
- **The onset-only design cannot tell fragmentation from rhythm.** Both (b) and (c) give
  an onset-locked peak near 105–120 ms, close to the real burst in lag and size.
- **The at-risk design recovers the truth.** Under (a) no history block gains; (b) is
  read as end-locked, (c) as onset-locked at the right lag (attenuated), and (d) in the
  right direction, anchor and lag, with no coupling the other way.

## Limits that hold for the real fits

1. Onset-locked history shorter than the previous event's duration is not estimable:
   those bins are not at risk. The "refractory period" of the onset-only design is the
   event itself.
2. A dependence within one bin (15 ms) of an event's end is not resolved; the end-locked
   kernel peaks one bin early.
3. With near-constant durations, an end is an onset shifted by a fixed lag and the two
   anchors cannot be told apart. In the simulation they separated when the durations'
   log-SD was about 0.2 or more and the canonical correlation of the onset and end blocks
   about 0.9 or less (`duration_aware_history.onset_offset_collinearity` measures it on a
   design); below that, history should be reported only as the joint onset-plus-end
   effect, without a mechanism.
4. One session is too sparse for per-lag kernels, so kernels are read pooled over each
   animal's sessions (`duration_aware_history.fit_pooled_history`) and only at lags with
   enough expected onsets (`duration_aware_history.lag_support`).
5. Not simulated: rate differences between stops (which create positive history in any
   design), a within-stop time course, the real duration distributions, and kernels that
   are not log-additive.

## What it means today

Family B is fit on the at-risk design, with self-history counted from event ends
(since 2026-11-03; [../rate_models.md](../rate_models.md), "At-risk bins"). The real
duration-aware refit's values are the `history_*` keys of `paper/reported/behavior.json`,
listed under their question in [../../paper/reported/questions.md](../../paper/reported/questions.md).
Where the real streams fall on limit 3 (their duration log-SD and the onset/end
canonical correlation) is not computed in a summary.
