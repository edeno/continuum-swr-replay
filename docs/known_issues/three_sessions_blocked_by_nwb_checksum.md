# Three sessions blocked by an upstream NWB checksum failure

**Status:** resolved on 2026-08-19. All 218 analysed epochs are in every table. The
cohort is 218, not 215. Found resolved by the exclusion audit of 2026-10-05
(`.claude/research/refresh_exclusions/exclusions_report.md`, section 3.8). The
cohort is now the epoch register's ([../epoch_register.md](../epoch_register.md)),
where these three epochs are `analysed`.

## What it was

From 2026-08-17 to 2026-08-19, three sessions were absent from the canonical
replay-content tables:

- the tables: `event_table`, `event_arm_table`, `event_patch_table`,
  `event_sequence_table`, `stop_table`, and the qvals join;
- the count: **215 of 218** sessions.

| animal | day | epoch |
|---|---|---|
| j16 | 20210707 | 12_r6 |
| senor | 20201102 | 02_r1 |
| wilbur | 20210331 | 12_r6 |

`prepare_data` pulled each session's `trials` and track graph from Spyglass. That
downloaded the source analysis NWB through DataJoint, and for these three the
download failed DataJoint's checksum, for example:

`'/stelmo/nwb/analysis/j1620210707_0VZUCNY9K0.nwb' downloaded but did not pass checksum`

(`logs/regen_noncache_20260817.log`). The build recorded the failure and skipped
the session.

## How it was resolved

On 2026-08-19, between 21:04 and 21:05, users `alison` and `sambray` rewrote the
three analysis files:

- `j1620210707_0VZUCNY9K0`;
- `senor20201102_5JRRG78UDM`;
- `wilbur20210331_7EIP36CASJ`.

Their hash (`uuid_from_file`) now matches `~external_analysis.contents_hash`. All
three epochs are decoded, by both the April decode and the October 2026 re-decode,
and have rows in all seven canonical tables:

- `event_table` has 157, 309 and 118 rows;
- `qvals_table` covers them through `stop_id`;
- so do the ContFrag tables.

`docs/audits/2026-08-25-covariate-regen-staging-audit.md` recorded the recovery in
the Replay and ContFrag tables.

## What no longer applies

- Every count is over the 218 analysed epochs, not 215.
- A parity gate that reports a 218→215 drop for these keys is now reporting a real
  regression, not the expected state.
- `model_specs.py`'s note that "215 of 218 had fitted before" describes fits made
  before the recovery.
