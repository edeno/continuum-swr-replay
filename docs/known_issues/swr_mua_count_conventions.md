# Convention: SWR/MUA counts differ between stop-level and bin-level tables

**Status:** documented convention (not a bug) — a rule to prevent misuse
**Found:** 2026-07-04 data-provenance / regression-integrity audit (§3)
**Severity:** low if the rule below is followed; a silent bias if the two counts
are cross-substituted in an exposure/offset term.

## What

The per-stop `n_swr_events` / `n_mua_events` counts (stop-level tables — `stop_table.parquet`) do **not**
exactly equal the timebin/onset counts of SWR/MUA events used by the per-bin (Family B)
and onset tables. Across the 218 HPC sessions the stop-level totals are slightly **larger**:

| count | stop-level rule | vs bin/onset | Δ |
| --- | --- | --- | --- |
| `n_swr_events` | inclusive per-timebin `is_swr_stop` over the **full** bout | onset-in-window | **stale — see below** |
| `n_mua_events` | inclusive per-timebin rule over the full bout | onset-in-window | **stale — see below** |
| `n_selected_events` (Replay source) | non-local event onset in the stop | same event population | **0 (matches exactly)** |

> **⚠️ The Δ magnitudes are stale (2026-08-12).** The previously recorded
> +~0.2 % (Δ82) / +~0.75 % (Δ250) were measured before the
> `unified-event-immobility` change, when SWR/MUA labels were **not**
> speed-gated. Both grains now draw from the same whole-event
> speed-eligible cohort (`analysis/hippocampus/event_immobility.py`), so
> both sides of the comparison moved. **Re-derive these numbers before
> citing them.** The *direction* of the effect and the rule below are
> unchanged: the two rules still differ, so the counts still must not be
> cross-substituted.

> **SWR source (2026-10-09).** Both grains now read their SWR labels from one source,
> the direct ripple_detection 2.0 Kay inventory
> (`analysis/hippocampus/swr_source.py`; design note
> [../design/2026-10-09-swr-source.md](../design/2026-10-09-swr-source.md)), in place
> of Spyglass's stored `RippleTimesV1` events. The per-bin label convention is
> unchanged (closed intervals, whole-event speed gate, source event numbers), and so
> are the per-grain rules below. Any re-derived Δ must be measured on the new source.

> **Artifact mask (2026-10-09, Task 18).** With an artifact run configured
> (`swr_source.ARTIFACT_INTERVALS_RUN_ID`), both grains honour the one artifact mask
> (`analysis/hippocampus/artifacts/masks.py`), and the rule below still holds:
>
> - **Counts.** An SWR, MUA or non-local event that shares any time with the mask
>   (closed intervals) is in neither grain: it is not an SWR, not a non-SWR event,
>   not counted. A decoded event (non-local; a ContFrag label; a Widloski event)
>   within 0.3 s of the mask (`event_eligibility.DECODER_REACH_S`) is dropped too. The
>   drop is made once, where the decode frame is read (`swr_source.with_swr_source`),
>   so the two grains drop the same events; `swr_source.artifact_event_drops` gives
>   each one's reason. The kept events keep their numbers and their `event_id`.
> - **Exposure.** The two time terms lose the mask's time, by different rules. A
>   stop's `immobility_duration_s` is its span (`stop_end - stop_start`) less exactly
>   the masked seconds inside it, so it is no longer the stop's span. A per-bin row
>   (Family B's design, the dense `stop_timebin` table, a PFC fit bin) is dropped
>   whole when it touches the mask, and a Family B bin also when its history window
>   (`model_specs.longest_history_window_s`, 0.15 s) reaches it. The per-bin exposure
>   therefore loses a little more than the stop-level one (the partial bins at the
>   mask's edges and the history margin): one more reason the two grains' terms are
>   not interchangeable.
> - **Not in any exposure:** the 0.3 s on each side of the mask in which decoded
>   events are dropped. Near a mask a decoded-event rate is therefore slightly
>   under-counted; on Task 16's scratch run the reach dropped 5 non-local events
>   in 22 epochs (389 were dropped for overlapping the mask).
>
> With no artifact run configured, every table and design is unchanged, byte for
> byte (checked on the 5 pilot epochs against `a108373`).

## Why

The stop-level `is_swr_stop` / MUA flags use the **inclusive per-timebin** rule over the
entire immobility bout (any timebin of the bout overlapping a SWR/MUA event counts), while
the bin/onset tables count an event by its **onset falling inside the analysis window**.
Events whose extent **straddles a bout/window boundary** are counted by the inclusive
stop-level rule but not by the onset-in-window rule, so the stop-level total is slightly
higher. Replay's `n_selected_events` is defined identically on both sides and matches exactly, so it is
the safe cross-table count.

This is the same dual convention already flagged in `CLAUDE.md`'s "Canonical per-event
classification table" note: the stop-level `is_swr_stop` (inclusive per-timebin) and the
event-level `is_swr` (strict positive-duration overlap) are deliberately different flags.

## The rule

`n_swr_events` / `n_mua_events` are event **counts** — they enter a rate model as the response
(numerator), never as the offset. The offset/exposure is always a **time** term
(`log(immobility_time)` per stop, `log(exposure)` seconds per bin), so this is not about
substituting a count into an offset.

**Do not cross-substitute the `n_swr_events` / `n_mua_events` count between the stop-level and bin-level
tables** — e.g. do not use the stop-level `n_swr_events` as the response for a bin-level model
(or compare a stop-level count to a bin-level count as if they were the same quantity).
The two differ because of boundary-straddling events, so mixing them biases the rate.
(The magnitude of that difference needs re-deriving — see the note above.)
When a count must be shared or compared across the two grains, use
Replay's `n_selected_events`, which is defined identically on both sides and matches exactly.
