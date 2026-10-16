# Three sessions blocked by an upstream NWB checksum failure

**Status:** resolved on 2026-08-19. The three epochs are in the cohort and in every table.
Found resolved by the exclusion audit of 2026-10-05
(`.claude/research/refresh_exclusions/exclusions_report.md`, section 3.8). The
cohort is the epoch register's ([../epoch_register.md](../epoch_register.md)),
where these three epochs are `analysed`.

**Cohort size, as this note has stated it:**

| from | cohort | status |
|---|---|---|
| (until 2026-08-19) | 215 | superseded 2026-08-19: the three epochs rejoined |
| 2026-08-19 | 218 | superseded 2026-10-11: the full pipeline refresh swapped in the R11 decodes and the register switched to 242 analysed epochs (255 decoded, less 13 `SUSTAINED_NOISE`; [known issue](2026-10-12-full-redecode-r11.md)) |
| 2026-10-11 | **242** | current |

The text below is the history of the 215-to-218 change. Where a count has been superseded, it
says so.

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
three epochs are decoded, by the April decode, by the first October 2026 re-decode and by
the R11 decode. As of 2026-10-05, on the 218-epoch tables, they had rows in all seven
canonical tables:

- `event_table` had 157, 309 and 118 rows (**superseded 2026-10-11**; the 242-epoch table
  has 167, 308 and 111, read from `replay_content/event_table.parquet` on 2026-10-16);
- `qvals_table` covered them through `stop_id`;
- so did the ContFrag tables.

The refresh audit of 2026-10-12 passed all 17 of its checks
([report](../audits/2026-10-12-full-pipeline-refresh.md), section 1). For the three epochs,
only the `event_table` counts were re-read; `qvals_table` and the ContFrag tables were not
re-counted.

`docs/audits/2026-08-25-covariate-regen-staging-audit.md` recorded the recovery in
the Replay and ContFrag tables.

## What no longer applies

- Every count is over the register's analysed epochs: 218 from 2026-08-19, and **242 since
  2026-10-11** (not 215, and no longer 218).
- A parity gate that reports a 218→215 drop for these keys is now reporting a real
  regression, not the expected state.
- `model_specs.py`'s note that "215 of 218 had fitted before" describes fits made
  before the recovery.
