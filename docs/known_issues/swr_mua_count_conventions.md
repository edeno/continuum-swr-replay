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
| `n_SWR` | inclusive per-timebin `is_swr_stop` over the **full** bout | onset-in-window | **+~0.2 % (Δ82)** |
| `n_MUA` | inclusive per-timebin rule over the full bout | onset-in-window | **+~0.75 % (Δ250)** |
| `n_non_local` | — | — | **0 (matches exactly)** |

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

**Do not cross-substitute `n_SWR` / `n_MUA` between the stop-level and bin-level tables in
an exposure/offset term.** A per-bin rate model's offset must use the bin-level onset
count (or `exposure` seconds); a per-stop model's offset uses the stop-level count. Mixing
them injects a ~0.2 %/0.75 % multiplicative bias into the rate. When a count must be shared
across the two grains, use `n_non_local`, which is identical on both sides.
