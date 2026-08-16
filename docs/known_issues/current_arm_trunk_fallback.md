# `current_arm` can name an arm the animal has not reached yet

**Status:** fixed in `event-location-context`; regeneration pending
**Found:** 2026-08-10, following the replay-tables review
**Severity of the defect:** negligible pooled (0.93 % of events *mislabelled*);
material per-animal for wilbur (3.9 %) and disqualifying for one session
(`wilbur20210329_16_r8`, 40.8 %)
**Footprint of the fix:** ~11 % of events lose a defined `current_arm` — an
order of magnitude larger than the defect, and a different population. See
[Footprint of the fix](#footprint-of-the-fix) before budgeting for it.

## Footprint of the fix

The severity line above counts events whose `is_current` was *wrong*. It is not
the number of events the fix changes, and quoting it as such understates the
change by roughly 12×.

`current_arm` is now defined only when the whole event window is at one well.
"Every bin at the same well" is a far stronger condition than "the modal segment
is a leaf arm", so it nulls a much larger set:

| | share of events |
| --- | --- |
| `current_arm` NaN, old rule | 0.00 % |
| `current_arm` NaN, new rule | **9.10 %** |

**Measured on the regenerated tables** (2026-08-16, all 218 sessions / 65,913
non-local events): 90.90 % `at_well`, 8.88 % `in_transit`, 0.21 % `mixed`.
`is_current` is NaN on exactly 9.10 % of event-arm rows, matching the off-well
event share. Per-animal 4.92 % (senor) → 11.64 % (chimi); per-session median
5.14 %, p90 15.50 %, max 58.13 % (`wilbur20210329_16_r8`), with 5 of 218
sessions entirely at-well.

The event and stop cohorts are unchanged by the rebuild — 65,913 events,
593,217 event-arm rows and 107,400 stops both before and after — so this is a
column refresh, not a renumbering, and the change is purely subtractive.

*(Pre-regeneration estimates of 10.8 % over 36 sessions, 12.4 % over 15, and
6.4 % over the seven cached j16 sessions are superseded by the line above. They
replayed the production path over stale event windows on session subsets; the
per-animal spread in particular was much wider in those samples — 3.8–25.2 % —
than in the full corpus.)*

**Most of it is not the trunk defect.** 77.5 % of the newly-NaN events had a
modal *leaf* arm under the old rule, so they were never mislabelled by the
trunk fallback this document describes. They are nulled because the animal sat
immobile on a leaf arm before `goal_arrival_time` — a correct application of
the new definition, but not a correction of a prior error.

`is_current` and `is_same_patch` propagate that NaN (see `eq_indicator_or_nan`),
so these events are dropped by complete-case fitting rather than entering a
model as "not the current arm" on all nine arms.

## Pre-fix measurements (superseded)

The text and measurements below describe the pre-fix artifacts. The canonical
builders now label the complete event/bout window as `at_well`, `in_transit`,
`mixed`, or `unknown`, preserve the modal observed segment as `physical_arm`,
and set `current_arm = location_arm` only for a window wholly at one well.
In-progress trips no longer expose their destination or outcome through past
context. Existing pickles retain the old values until regeneration.

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

## Resolution

- Trial history now advances at `goal_arrival_time` (with the established
  per-row `end_time` fallback), not departure.
- Exact arrival is at-well; exact departure is in-transit.
- Stop rows carry a nullable, zero-based `completed_trial_id` plus
  `completed_trial_arm`. The ID is regenerated in completion order within each
  session; repeated immobility bouts after one arrival share the same ID.
- `recent_switch_rate` and `leaky_q_value_diff` are computed once from the full
  completion-ordered trial stream, then stored once per stop context. Model
  consumers broadcast them to events by `stop_id`. Thus a completed trial with
  no immobility bout still contributes exactly once to later history.
- No eligible event is dropped from the *tables* by default; the location
  columns are retained so every cohort decision is explicit and auditable.
  Models are a different matter: `is_current` is NaN off-well, so any spec
  containing it fits the at-well cohort. That is the only cohort in which
  `is_current` is defined — it is not a "sensitivity" subset.
- `current_arm` is NaN for transit/mixed/unknown windows instead of guessing a
  destination leaf; `physical_arm` remains available for topology and QC.

## Guidance for legacy artifacts

- **Do not treat the legacy pooled estimates as validated under the new
  definition.** The specific destination-mislabel defect affected 0.93 % of
  events and is therefore unlikely to dominate a pooled effect, but that is a
  plausibility argument rather than a result comparison. The new whole-window
  location rule changes a much larger 10.8 % cohort, so only regeneration and
  an old-versus-new coefficient audit can establish conclusion stability; see
  [Footprint of the fix](#footprint-of-the-fix).
- **Sensitivity-check wilbur** if per-animal estimates are quoted.
- **Exclude `wilbur20210329_16_r8`** from per-session analyses, or report it with
  the caveat. It is compromised from two independent directions.
- Regenerate canonical tables before using the new location labels or recency
  semantics. Until then, retain the legacy sensitivity cautions above.

## Reproducing

The measurement reads only the pre-saved `data_over_time` pickles and the cached
`trials` frames (`replay_content/_session_cache/`), so it needs no Spyglass access
and takes ~9 s per session. It replays `build_event_table`'s own rule — modal
`track_segment_id` → `arm_to_well` → `current_well` fallback — and compares each
event's midpoint against `_goal_arrival_time` from `add_trial_context`.
