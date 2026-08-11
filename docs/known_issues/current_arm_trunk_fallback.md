# `current_arm` can name an arm the animal has not reached yet

**Status:** confirmed defect, measured, not fixed
**Found:** 2026-08-10, following the replay-tables review
**Severity:** negligible pooled (0.93 % of events); material per-animal for wilbur
(3.9 %) and disqualifying for one session (`wilbur20210329_16_r8`, 40.8 %)

## What

`build_event_table` sets `current_arm` to the modal `track_segment_id` during the
event — but only when that segment touches a well. Trunk arms (6/7/8) connect two
junctions and no well, so `arm_to_well` returns NaN and the builder falls back to
the arm of `current_well`
([`tables.py`](../../src/continuum_swr_replay/analysis/replay/tables.py), the
`_single_arm_from_well(_strict_first(g["current_well"]), ...)` branch).

`current_well` is `to_well` of the latest trial **started** at or before the event
(`add_trial_context`), and a trial spans departure → arrival. So when the animal is
paused mid-transit on a trunk arm, `current_arm` is the arm it is *heading toward*,
not where it is. `is_current` — the pivotal predictor in every content model — is
then set on an arm the animal has not reached.

This is the same root cause as the `is_after_switch` timing defect (C1 in the
2026-08-08 audit): trial context indexed by departure rather than arrival.

## How often (all 218 HPC sessions, 65,913 events)

| | events | share |
| --- | --- | --- |
| trunk fallback fires | 640 | 0.97 % |
| …and the trip is still in progress | **616** | **0.935 %** |
| …modal segment was a trunk arm | 640 | 100 % |
| …assigned arm is NaN | 0 | — |

The fallback never fires for a NaN modal segment, so every case is the transit
situation described above.

**It is not uniformly distributed.** 69 of 218 sessions see the fallback fire at
all, 68 of them with the trip still in progress:

| animal | leaking events | share of that animal's events |
| --- | --- | --- |
| wilbur | 278 | **3.88 %** |
| chimi | 195 | 0.87 % |
| j16 | 97 | 0.68 % |
| senor | 45 | 0.25 % |
| peanut | 1 | 0.02 % |

A 165× spread across animals. Per-session share: median 0 %, max **40.75 %**.

## The one session that matters: `wilbur20210329_16_r8`

218 of its 535 events (40.8 %) get a lookahead `current_arm`. Diagnosis:

- 26.0 % of its **immobile** timebins are on a trunk arm. Across all 218
  sessions that statistic has median 0.25 % and p90 1.4 %; the next-highest
  session is `j1620210707_06_r3` at 10.4 %, and five sessions exceed 4 %. So
  this session is ~106× the corpus median but only 2.5× the runner-up — it is
  the extreme of a continuum, not a lone anomaly.
- The leak is concentrated at the end: **96 % of the 218 events fall in the last
  two deciles**, 78 % in the final decile — where deciles divide the span from
  the first to the last event midpoint. Dividing the full recording instead
  gives 94 % and 77 %; dividing into equal-count event deciles gives 49 % and
  24 %.
- The arm assigned is one the animal reaches a **median 46 s later** (p90 103 s,
  max 146 s). 89 % are assigned more than 10 s before arrival.
- The behavior is real, not a tracking artifact: during the longest pause the
  linear position varies over **0.9 cm** across 136 s at a median speed of
  0.63 cm/s. Position is genuinely stationary, mid-trunk. Frozen-sample rate —
  defined as raw `head_position_x` *and* `head_position_y` both unchanged from
  the previous sample — is 9.7 % for leaking events versus 10.0 % for the rest,
  so frozen tracking does not explain which events leak. (Under
  `linear_position` instead the two are 10.0 % and 72.1 %, which points the same
  way even more strongly; the definition matters, so state it when quoting the
  number.)

So the animal made one very slow, heavily paused final approach; non-local events
fired during the pauses; and each was labelled with the destination it would reach
up to 2.4 minutes later.

This is the same session the audit's C8 flags as losing 33 % of its events to the
value-block join (one of the eight "rat quit early" sessions). Both flags share a
cause: **behavioral engagement ended before the recording did, while the event
pipeline kept producing events.**

## What to do

- **Pooled estimates are safe.** 0.93 % of events with a mislabelled `is_current`
  cannot move an effect of the size `is_current` carries.
- **Sensitivity-check wilbur** if per-animal estimates are quoted.
- **Exclude `wilbur20210329_16_r8`** from per-session analyses, or report it with
  the caveat. It is compromised from two independent directions.
- **The real fix** is the same one C1 needs: index trial context by *arrival*, not
  departure, so a trip in progress does not contribute its own destination. Until
  then, an event whose modal segment is a trunk arm should arguably carry
  `current_arm = NaN` rather than a guess.
- Deciding the **well-versus-transit event scope** would remove the issue by
  construction: restricting the event population to well stops drops all 616.

## Reproducing

The measurement reads only the pre-saved `data_over_time` pickles and the cached
`trials` frames (`replay_content/_session_cache/`), so it needs no Spyglass access
and takes ~9 s per session. It replays `build_event_table`'s own rule — modal
`track_segment_id` → `arm_to_well` → `current_well` fallback — and compares each
event's midpoint against `_goal_arrival_time` from `add_trial_context`.
