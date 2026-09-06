# Three sessions blocked by an upstream NWB checksum failure

**Status:** external, outside our control. Proceed with the 215-session cohort;
do not gate on recovery.

## What

Three sessions are absent from the canonical replay-content tables
(`event_table`, `event_arm_table`, `event_patch_table`, `event_sequence_table`,
`stop_table`, and the qvals join), leaving **215 of 218** sessions:

| animal | day | epoch |
|---|---|---|
| j16 | 20210707 | 12_r6 |
| senor | 20201102 | 02_r1 |
| wilbur | 20210331 | 12_r6 |

`event_onset_table` still carries all 218 (it is built by a lighter pipeline
that does not hit the failing fetch), which is why
`attach_is_swr_to_onsets` scopes to the shared sessions and warns.

## Why

`prepare_data` pulls each session's `trials` and track graph from Spyglass even
though the decodes are cached, so it downloads the source NWB via DataJoint.
For these three the download fails a **checksum** verification —
e.g. `'/stelmo/nwb/analysis/j1620210707_0VZUCNY9K0.nwb' downloaded but did not
pass checksum` (see `logs/regen_noncache_20260817.log`). `prepare_data`'s broad
`except` records the failure and skips the session, so the table build completes
with 215.

This is an **upstream data-store issue we do not control.** It may be repaired
eventually (a re-ingest / checksum reset on the Spyglass side), but the analysis
must not depend on that.

## Decision

- The canonical cohort is **215 sessions**. Every downstream count, figure, and
  claim is stated over 215, not 218.
- The harmonization parity gate correctly reports the 218→215 drop; that FAIL is
  the *expected* state for these three keys, not a regression. Do not "fix" it by
  weakening the gate — recognise these three keys specifically.
- If the upstream checksums are later resolved, regenerate the atomic set to pick
  the three back up (they need no code change, only a clean fetch).

## Related

- `attach_is_swr_to_onsets` session-scoping + warning (analysis/replay/per_bin_design.py).
- `model_specs.py` already annotates "215 of 218 had fitted before".
