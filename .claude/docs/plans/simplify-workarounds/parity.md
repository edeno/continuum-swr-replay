# Table parity for the simplify-workarounds branch

Every task on this branch must leave every canonical table unchanged. This file
gives the commands that check it. Task 0 wrote it, at `155db501` (master
`99034aab` plus the plan commit; no pipeline code changed).

The tool is `scripts/qa/table_parity.py`:

- `regen` rebuilds every canonical table into a new root. That is the replay
  tables (stop, qvals, event, event-arm, event-patch, event-sequence,
  event-onset, stop-timebin) and the ContFrag SWR and MUA tables. It uses the
  builders, flags and order of the 2026-10-11 rebuild the live tables come from
  (`.claude/research/refresh_r13/r13c_report.md`): stop bootstrap, then qvals,
  then one replay shard per animal beside the time-bin build, then assemble,
  then ContFrag SWR and MUA.
- `diff` compares two roots table by table:
  - values on each table's row key, exactly (`tables.parity.diff_table`,
    `atol=0`);
  - the column list and order, every column's dtype, and the on-disk row order;
  - the tables' attrs, and the two standardization references byte for byte.

  It reports every missing table, every extra `*.parquet`/`*.json` in the
  candidate's table folders and every differing table, then exits 1 if there
  is any. `--restrict` takes its sessions from the candidate slice's own
  `epoch_register.csv` and its qvals stops from the reference's stop table, so
  a candidate that lost a session cannot shrink the reference.

## Slice parity (each task)

The slice is three epochs:

| Epoch | Animal | Why | Events | Stops |
|---|---|---|---|---|
| `chimi20200217_06_r3_HPC` | chimi | 21 excluded intervals (24.1 s), so the analysed-time drops run | 385 | 297 |
| `j1620210712_10_r5_HPC` | j16 | one excluded interval; one SWR dropped as moving | 217 | 525 |
| `wilbur20210330_04_r2_HPC` | wilbur | has mPFC/OFC decodes; 4 excluded intervals | 114 | 418 |

The frozen reference is `Processed-Data/_scratch_simplify_20261024/slice_reference/`.
It is read-only and its `SHA256SUMS` lists every table file.

Run from the repo root. Replace `<task>` with a fresh name such as `t09`;
`regen` refuses a root that already exists.

```bash
R=Processed-Data/_scratch_simplify_20261024
uv run python scripts/qa/table_parity.py regen --root $R/slice_<task> \
    --sessions chimi20200217_06_r3_HPC j1620210712_10_r5_HPC wilbur20210330_04_r2_HPC \
  && uv run python scripts/qa/table_parity.py diff \
    --candidate $R/slice_<task> --reference $R/slice_reference
```

It takes about 7 minutes. qvals takes about 4 of them, almost all of it the
per-animal DataJoint reads. A pass ends with `0 difference(s)` and exit 0.

Optional byte-level check (Task 0 found it byte-identical too):

```bash
(cd $R/slice_reference && sha256sum -c --quiet SHA256SUMS) # checks the reference itself
(cd $R/slice_<task> && sha256sum -c --quiet ../slice_reference/SHA256SUMS)
```

The ContFrag tables, the tables' attrs and the two standardization references
are not covered by `compare_to_golden_baseline.py`. `table_parity.py diff` checks
all of them, so use it.

How the slice works. Every builder takes its sessions from the epoch register.
So `regen --sessions` writes `<root>/epoch_register.csv`, a copy of the register
in which every other analysed epoch is marked `excluded` (reason
`MISSING_INPUT`, evidence "outside the table-parity slice"). It then runs each
builder in a child process with `epoch_register.REGISTER_PATH` pointed at that
copy. The builders and their code paths are the full cohort's.

What the root holds and reads:
- `replay_content/` and `contfrag_replay_content/` (the layout of
  `Processed-Data`), plus `logs/<step>.log` and `timings.json`.
- Its own copy of the needed `_session_cache/comrie_photogate_trials_v4` entries,
  so nothing is written to the live cache. The cache is frozen, so slice parity
  (and the full run) do not cover changes to the code that builds it, which is
  `load_behavioral_context`'s Spyglass path (`load_data`'s trials and track
  graph). A task that changes that code needs its own check. A cache entry
  missing from the canonical cache is logged as a warning, and that session falls
  back to Spyglass and is cached into the root.
- Decodes and the artifact/SWR runs are read from `paths.processed_data_dir()`
  as usual (since Task 19 every data root is resolved when called from
  `CSR_PROCESSED_DATA_DIR`, else the checkout's `Processed-Data/`; the commands
  above are unchanged). `CSR_PROCESSED_DATA_DIR` must be unset for `regen`:
  pointing it at a scratch root would hide the decodes.

Ruling R2 (Task 19): if the data-root mechanism changes, keep these commands
working or update them here.

## Full-cohort parity (Task 21)

Use tmux session `simplify_parity`:

```bash
tmux new-session -d -s simplify_parity 'cd /cumulus/edeno/continuum-swr-replay && \
  uv run python scripts/qa/table_parity.py regen \
    --root Processed-Data/_scratch_simplify_20261024/full_t21 \
    > Processed-Data/_scratch_simplify_20261024/full_t21.log 2>&1; \
  echo EXIT=$? >> Processed-Data/_scratch_simplify_20261024/full_t21.log'
# when EXIT=0 appears:
uv run python scripts/qa/table_parity.py diff \
    --candidate Processed-Data/_scratch_simplify_20261024/full_t21 \
    --reference Processed-Data
uv run pytest -m real_data -p no:cacheprovider
```

The reference is the live tables, `Processed-Data/replay_content/` and
`Processed-Data/contfrag_replay_content/`, built at `bb19449` on 2026-10-11
(`tables_manifest.json`). Task 0 showed they are a valid reference for the
pre-branch code:

1. The slice reference was compared with the live tables restricted to its three
   epochs (`diff --restrict --reference Processed-Data`). 14 of the 16 tables are
   identical in every row, key and column, and also in column order, dtypes and
   row order (checked again after fix round 1). That covers all the ContFrag tables
   and the stop, event, event-patch, event-sequence, event-onset and stop-timebin
   tables.
2. Only the pooled `_z` columns differ, and that is expected. These are the 4
   event-arm `_z` columns, the 6 qvals `_z` columns and the two standardization
   references. Their moments are pooled over the cohort, and on the slice that
   cohort is three epochs. When the slice's raw columns are standardized with the
   live references' moments (`attach_standardized(..., reference=live)`), all 10
   `_z` columns equal the live ones exactly (max |diff| 0, NaN positions equal).
3. The attrs are equal except `stop_timebin_table`'s
   `onsets_in_excluded_bins_by_event_type`. That is a sum over the cohort: SWR 1
   in the live table, 0 in the slice.

A full regeneration therefore has to match the live tables exactly, including
the pooled columns and that attr.

Cost. The one-epoch `regen` (chimi20200217_06_r3) took 191 s in total. Of that,
qvals took 97 s (mostly the fixed DataJoint read), the stop step 15 s, the time-bin
build 21 s, the shard 39 s, assemble 8 s, and each ContFrag build 32 s (the two run
side by side). The three-epoch slice took 428 s and 419 s on two runs.

For the full cohort, `regen` reproduces R13c's parallel layout. R13c measured
these times at `bb19449` with the same builders:

| Step | R13c time |
|---|---|
| stop | 9:41 |
| qvals | 11:23 |
| 5 shards in parallel (the longest is j16) | 1:14:53 |
| time-bin, beside the shards | 28:41 |
| assemble | 0:58 |
| ContFrag SWR and MUA in parallel | 1:36:58 each |

That is about **3 h 15 min** of wall time. Peak RSS is about 12 GB per shard or
ContFrag build, so about 70 GB at the widest step. The root needs about 2 GB of
disk, including the shards.

The per-epoch figures say the same. One small epoch costs about 140 s across the
steps other than qvals. Run serially over 242 epochs that would be about 9 h. The
parallel layout cuts it to the figure above.
