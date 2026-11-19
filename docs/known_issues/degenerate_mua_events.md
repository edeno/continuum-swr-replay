# Data-quality flag: degenerate multi-second MUA events

**Status:** open — needs upstream investigation
**Found:** 2026-06-05, while choosing the ContFrag event-immobility rule
**Severity:** low for current analyses (worked around), but points at a
possible bug in MUA event labeling.

## What

A small number of `MUA_event` labels in the ContFrag `data_over_time`
frames span **multiple seconds** — far longer than a plausible
multi-unit-synchrony burst. Measured across all 218 HPC ContFrag sessions
(`*_contfrag_data_over_time.pkl`):

| detector | n events | median dur | p99 dur | max dur | events > 1 s |
|----------|---------:|-----------:|--------:|--------:|-------------:|
| `SWR_event` | 38,824 | 78 ms | 300 ms | 1,754 ms | 3 (0.008%) |
| `MUA_event` | 33,482 | 112 ms | 408 ms | **23,127 ms** | 58 (0.173%) |

The longest MUA "event" is **~23 s** (in
`senor20201101_14_r7_HPC`). Such events have immobile *endpoints* but
contain real locomotion in the middle (interior mobile gaps up to ~4 s) —
i.e., a single MUA label persists across a stop → run → stop sequence.
The labels themselves are contiguous in time and uniquely numbered, so
this is not a grouping artifact; the detector/labeling is genuinely
emitting multi-second MUA events.

SWR events do not show this (max 1.75 s, a single outlier; only 3 events
> 1 s), so the issue is specific to MUA.

## Why it matters

A 23 s "MUA event" is almost certainly malformed — a real MUA burst is
tens of milliseconds. These events:
- inflate any per-event MUA content/duration summary,
- span multiple behavioral states (immobility + locomotion), so they are
  ill-defined as "replay during a stop."

## How it's currently handled (workaround, not a fix)

The ContFrag event-table builders use the **all-immobile ("max")**
immobility rule: an event is included only if *every* timebin is below the
immobility threshold. Movement-spanning events (which is exactly this
degenerate set) are therefore **excluded** from the ContFrag tables, so
they do not enter the SWR/MUA replay-content fits or the event-source
comparison. This makes the downstream analysis robust to the issue, but
does **not** fix the underlying labeling.

## To investigate

- Where MUA event intervals are produced upstream (the MUA / high-synchrony
  detector feeding `get_labels` for the ContFrag pipeline). Check the
  burst-merging / minimum-separation / maximum-duration logic.
- Whether a maximum-event-duration cap (or a re-split on movement) should
  be applied at detection time.
- `senor20201101_14_r7_HPC` is a concrete reproducer (23 s MUA event).

Reproduce the per-session duration scan with a short script that segments
each `MUA_event` label into contiguous runs and reports run durations
(see the git history around 2026-06-05 for the throwaway measurement).
