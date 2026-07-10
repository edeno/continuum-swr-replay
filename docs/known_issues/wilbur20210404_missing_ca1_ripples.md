# Exclusion: wilbur20210404 has no CA1 ripple extraction

**Status:** documented exclusion (data-provenance gap) — 5 run epochs excluded
**Found:** 2026-07-10, follow-up to the branch code-review audit (F8 `get_ripple` review)
**Severity:** coverage loss (5 wilbur run epochs absent from all analyses); **not** a
wrong-number bug — the affected session simply never produced outputs.

## What

`wilbur20210404_.nwb` is the only project session whose sharp-wave-ripple times were
never computed under `group_name = "CA1"`. Its ripples exist **only** under
`group_name = "sb_wilbur_ripple"` — a different ripple extraction (different
channel/reference set) than the `CA1` extraction used for every other project session.

`get_ripple` (`src/continuum_swr_replay/data_loaders/lfp.py`) restricts on
`group_name = "CA1"`, so for this session the query matches **0 rows** and previously
died on the opaque DataJoint `fetch1` "0 tuples found" error inside `load_data`. Because
`run_decode` catches per-session exceptions and moves on, the 5 run epochs
(`02_r1, 04_r2, 06_r3, 08_r4, 10_r5`) were **silently dropped** from every
decode / replay / rate / content analysis.

Confirmed on the filesystem: `wilbur20210404` has **0** HPC decode result files, while all
six other wilbur dates decoded normally (44 result files).

## Why it is excluded (not recovered)

`sb_wilbur_ripple` is a distinct extraction. Falling back to it would put these 5 epochs
on a **different ripple definition** than the other 218 sessions, risking a systematic
non-comparability in every SWR-overlap-dependent result. Rather than mix definitions, the
session is excluded and documented. (The clean alternative — computing a matching CA1
`RippleTimesV1` extraction for `wilbur20210404` via the Spyglass ripple pipeline — would
let these epochs rejoin the analysis on equal footing; it was deferred.)

## Evidence (live DataJoint, 2026-07-10)

Across all project sessions under the loader's filter
(`filter_name = ripple_150_250`, `filter_sampling_rate = 1000`,
`ripple_param_name = default`):

| animal | group_name(s) present | CA1 rows |
| --- | --- | --- |
| chimi, j16, senor | `CA1` | all |
| peanut | `CA1` | all (no `CA1_ripple_peanut` for project files) |
| wilbur | `CA1` (6 dates) **+ `sb_wilbur_ripple` (wilbur20210404 only)** | 44 |

- 218 / 229 project session-intervals include a `CA1` row → `get_ripple` works.
- 11 / 229 (all from `wilbur20210404_.nwb`) have **only** `sb_wilbur_ripple` → excluded.
- No session has `CA1` *and* a parallel extraction, so the `group_name = "CA1"` restriction
  is unambiguous everywhere it matches.

## Current behavior

`get_ripple` now performs an explicit row-count check and raises a clear `ValueError`
naming the session and the `group_name`(s) actually present, pointing here. The failure is
caught and aggregated per-epoch by `run_decode`, so the exclusion is **explicit and logged**
rather than a silent opaque crash. The exclusion is also listed in the `NWB_FILES` docstring
in `src/continuum_swr_replay/data_loaders/constants.py`.

## To un-exclude

Populate `RippleTimesV1` for `wilbur20210404` with `group_name = "CA1"` (matching the other
sessions' ripple parameters), then re-run HPC decoding for its 5 run epochs. No code change
is needed once a CA1 extraction exists — `get_ripple` will match it and the count check
passes.
