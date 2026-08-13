# Convention: SWR/MUA counts differ between stop-level and bin-level tables

**Status:** documented convention (not a bug) — a rule to prevent misuse
**Found:** 2026-07-04 data-provenance / regression-integrity audit (§3)
**Severity:** low if the rule below is followed; a silent bias if the two counts
are cross-substituted in an exposure/offset term.

## What

The per-stop `n_SWR` / `n_MUA` counts (stop-level tables — `stop_table.pkl`) do **not**
exactly equal the timebin/onset counts of SWR/MUA events used by the per-bin (Family B)
and onset tables. Across the 218 HPC sessions the stop-level totals are slightly **larger**:

| count | stop-level rule | vs bin/onset | Δ |
| --- | --- | --- | --- |
| `n_SWR` | inclusive per-timebin `is_swr_stop` over the **full** bout | onset-in-window | **stale — see below** |
| `n_MUA` | inclusive per-timebin rule over the full bout | onset-in-window | **stale — see below** |
| `n_non_local` | — | — | **0 (matches exactly)** |

> **⚠️ The Δ magnitudes are stale (2026-08-12).** The previously recorded
> +~0.2 % (Δ82) / +~0.75 % (Δ250) were measured before the
> `unified-event-immobility` change, when SWR/MUA labels were **not**
> speed-gated. Both grains now draw from the same whole-event
> speed-eligible cohort (`analysis/hippocampus/event_immobility.py`), so
> both sides of the comparison moved. **Re-derive these numbers before
> citing them.** The *direction* of the effect and the rule below are
> unchanged: the two rules still differ, so the counts still must not be
> cross-substituted.

## Why

The stop-level `is_swr_stop` / MUA flags use the **inclusive per-timebin** rule over the
entire immobility bout (any timebin of the bout overlapping a SWR/MUA event counts), while
the bin/onset tables count an event by its **onset falling inside the analysis window**.
Events whose extent **straddles a bout/window boundary** are counted by the inclusive
stop-level rule but not by the onset-in-window rule, so the stop-level total is slightly
higher. `n_non_local` is defined identically on both sides and matches exactly, so it is
the safe cross-table count.

This is the same dual convention already flagged in `CLAUDE.md`'s "Canonical per-event
classification table" note: the stop-level `is_swr_stop` (inclusive per-timebin) and the
event-level `is_swr` (strict positive-duration overlap) are deliberately different flags.

## The rule

`n_SWR` / `n_MUA` are event **counts** — they enter a rate model as the response
(numerator), never as the offset. The offset/exposure is always a **time** term
(`log(immobility_time)` per stop, `log(exposure)` seconds per bin), so this is not about
substituting a count into an offset.

**Do not cross-substitute the `n_SWR` / `n_MUA` count between the stop-level and bin-level
tables** — e.g. do not use the stop-level `n_SWR` as the response for a bin-level model
(or compare a stop-level count to a bin-level count as if they were the same quantity).
The two differ because of boundary-straddling events, so mixing them biases the rate.
(The magnitude of that difference needs re-deriving — see the note above.)
When a count must be shared or compared across the two grains, use
`n_non_local`, which is defined identically on both sides and matches exactly.
