# Stop-table regeneration for the reward-history basis columns

**Date:** 2026-09-04
**Code:** `4fda746ba937ce67e400e13352d2c052acfa1d56` (`regression-pipeline-fixes-phase-1`)
**Artifact:** `Processed-Data/replay_content/stop_table.parquet`
**Backup of the pre-change table:** `Processed-Data/replay_content/_pre_fix_backup_20260904/stop_table.parquet`
**Candidate (staging):** `Processed-Data/replay_content/_staging_stop_regen_full_20260904/stop_table.parquet`

## Decision

**Promoted.** The regenerated stop table adds four columns
(`reward_history_rcos_0`, `reward_history_rcos_1`, `reward_history_rcos_2`,
`reward_history_incomplete`) and changes nothing else: all 49 pre-existing
columns are identical to the backup in dtype, value, and NaN position, over the
same 107,400 rows in the same `stop_id` order. The four other canonical tables
rebuilt on the way are byte-identical (SHA-256) to the live files and were not
promoted.

**The covariate schema version stamp is unchanged at
`covariate_contract_v5_dio_poke_trials`.** Adding columns is not a vocabulary
change: the stamp records the behavioral-derivation contract (the DIO-poke trial
definition and the canonical names/row keys), and none of that moved. The
promoted table carries the identical `attrs` dict as the backup and passes
`require_covariate_schema(table="stop")`.

## Regeneration path and timing

`scripts/decoding/regenerate_stop_table.py` was rejected: it calls
`load_data()` directly, which requires a live Spyglass/DataJoint connection.
The rebuild instead ran from the behavioral-context cache
(`_session_cache/comrie_photogate_trials_v4`, 218 entries) through the canonical
driver, into a staging directory:

```bash
tmux new-session -d -s stop_table_regen \
  "/usr/bin/time -v uv run python -m continuum_swr_replay.analysis.replay.prepare_data \
     --from-cache \
     --output-dir      Processed-Data/replay_content/_staging_stop_regen_full_20260904 \
     --behavior-cache-dir Processed-Data/replay_content/_session_cache \
     --contingency-dir Processed-Data/replay_content \
     > logs/stop_regen/full.log 2>&1"
```

| run | scope | wall clock | peak RSS | exit |
|---|---|---|---|---|
| smoke, full driver (`--max-per-animal 1`) | 5 sessions | 3:38.68 | 11.75 GB | 0 |
| smoke, `--stop-only` (`--max-per-animal 1`) | 5 sessions | 0:19.19 | 3.16 GB | 0 |
| **full rebuild, tmux `stop_table_regen`** | **218 sessions** | **1:41:02** | **12.46 GB** | **0** |
| cross-check, `--stop-only` (all sessions) | 218 sessions | 5:56.33 | 3.82 GB | 0 |

Extrapolation from the smoke (42 s/session over 5 sessions) predicted ≈2.5 h for
218 sessions; the actual full run took 1 h 41 min (28 s/session) because the
smoke's per-animal-first sessions are among the larger ones. Peak RSS was
12.5 GB against 3.8 TB free, on a 96-core host with load average ≈1. No session
was skipped and no error was logged: the driver reported
`events: 65913, event_arm rows: 593217, stops: 107400, sessions: 218` and the
cross-table invariant `Σ n_selected_events == len(event_table) == 65913`.

Log: `logs/stop_regen/full.log` (smoke logs: `smoke_full.log`,
`smoke_stoponly.log`; cross-check: `stoponly_full.log`).

## Artifact integrity

| file | SHA-256 | rows |
|---|---|---|
| pre-change `stop_table.parquet` (= `_pre_fix_backup_20260904/`) | `a9954d6479b13e18fa470c68bb802b8e15419befcc5a67a452c63910381135c3` | 107,400 |
| promoted `stop_table.parquet` | `b261d9932163b34fdb604122f563fe9a42c1c7b88e5eb21b297b0999165b6b3e` | 107,400 |

The pre-change digest is the same one recorded for `stop_table` in
`_golden_baseline_v5_v4cache/manifest.json`, so the backup is the frozen golden
baseline byte-for-byte.

The other four canonical tables were rebuilt by the same run and are
**byte-identical to the live files**; none was promoted (the live files were
never touched):

| table | SHA-256 (live == rebuilt) | rows |
|---|---|---|
| `event_table` | `68ffdc1bdce6881cd53a6f55ae813020907cd6eb86860a3d90a07a699869491c` | 65,913 |
| `event_arm_table` | `6b11ce36bcf23fa2b0d81cd56519ba6973a196e2f512ca1fd3011b7f2cca83f3` | 593,217 |
| `event_patch_table` | `ceb357984b2f1dfe326851a6dffa9f901405358c068d8a30c0e4a511760eff0d` | 197,739 |
| `event_sequence_table` | `9f23c2b6071201c32434cf66492fe32112f43ef7f12f8006d84abcad73d7b8ed` | 2,971,614 |

That the whole rest of the pipeline reproduces bit-for-bit is the strongest
available evidence that the rebuild environment matches the one that produced
the live tables, and that the only thing the new code changed is the stop table.

## Parity of the pre-existing columns

Checked by `scripts/qa/audit_stop_table_reward_history.py`, comparing the
candidate against the backup positionally (rows are **not** re-sorted, so a
reordering would fail):

- rows: 107,400 vs 107,400
- `stop_id`: identical set **and** identical order (`Series.equals`)
- schema attrs: identical dict (see the stamp note above)
- columns: 53 vs 49; the candidate-only set is exactly the four added columns,
  and deleting them recovers the reference column order exactly (they are
  inserted after `known_reward_outcome_lag_10`, not appended)
- **49/49 pre-existing columns identical** in dtype, value, and NaN position

| column | dtype | dtype match | value mismatches | NaN-position mismatches | result |
|---|---|---|---|---|---|
| `animal` | object | yes | 0 | 0 | identical |
| `session_date` | int64 | yes | 0 | 0 | identical |
| `epoch` | object | yes | 0 | 0 | identical |
| `stop_idx` | int64 | yes | 0 | 0 | identical |
| `stop_id` | uint64 | yes | 0 | 0 | identical |
| `stop_start` | float64 | yes | 0 | 0 | identical |
| `stop_end` | float64 | yes | 0 | 0 | identical |
| `immobility_duration_s` | float64 | yes | 0 | 0 | identical |
| `reward_fraction_last10_known_trials` | float64 | yes | 0 | 0 | identical |
| `is_swr_stop` | bool | yes | 0 | 0 | identical |
| `n_swr_events` | int64 | yes | 0 | 0 | identical |
| `is_mua_stop` | bool | yes | 0 | 0 | identical |
| `n_mua_events` | int64 | yes | 0 | 0 | identical |
| `has_replay` | bool | yes | 0 | 0 | identical |
| `n_selected_events` | int64 | yes | 0 | 0 | identical |
| `did_last_completed_trip_switch_patch` | float64 | yes | 0 | 0 | identical |
| `is_rewarded_outcome` | float64 | yes | 0 | 0 | identical |
| `completed_trial_reward` | float64 | yes | 0 | 0 | identical |
| `completed_trial_id` | Int64 | yes | 0 | 0 | identical |
| `completed_trial_arm` | float64 | yes | 0 | 0 | identical |
| `patch_switch_fraction_last5_completed_trips` | float64 | yes | 0 | 0 | identical |
| `leaky_q_value_diff` | float64 | yes | 0 | 0 | identical |
| `is_stay` | float64 | yes | 0 | 0 | identical |
| `occupancy_state` | object | yes | 0 | 0 | identical |
| `at_well` | boolean | yes | 0 | 0 | identical |
| `occupancy_well_id` | float64 | yes | 0 | 0 | identical |
| `modal_observed_track_segment_id` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_min` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_mean` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_max` | float64 | yes | 0 | 0 | identical |
| `occupancy_well_arm` | float64 | yes | 0 | 0 | identical |
| `prev_arm` | float64 | yes | 0 | 0 | identical |
| `next_arm` | float64 | yes | 0 | 0 | identical |
| `last_rewarded_arm` | float64 | yes | 0 | 0 | identical |
| `current_patch` | float64 | yes | 0 | 0 | identical |
| `prev_patch` | float64 | yes | 0 | 0 | identical |
| `next_patch` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_1` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_2` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_3` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_4` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_5` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_6` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_7` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_8` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_9` | float64 | yes | 0 | 0 | identical |
| `known_reward_outcome_lag_10` | float64 | yes | 0 | 0 | identical |
| `n_nonlocal_swr_events` | int64 | yes | 0 | 0 | identical |
| `n_nonlocal_non_swr_events` | int64 | yes | 0 | 0 | identical |

The parity comparison has teeth: injecting a 1e-6 float perturbation, a changed
`completed_trial_arm` code, a flipped nullable-boolean `at_well`, a `+1` on an
`int64` count, a `+1` on a nullable `Int64` id, or a single extra NaN each made
the script raise. Integer columns are compared exactly rather than through a
float cast -- 107,301 of the 107,400 `uint64` `stop_id` values exceed 2**53, so
a float comparison would have been blind to a one-unit change in almost every
row; a `+1` on one `stop_id` is caught.

An independent `--stop-only --from-cache` build of all 218 sessions (a different
driver entry point, run in a separate process) agrees with the promoted table on
all 51 columns it produces — the two columns it deliberately omits are the
assembly-level `n_nonlocal_swr_events` / `n_nonlocal_non_swr_events` split. The
builder is therefore deterministic across runs, not merely self-consistent.

## The added columns

```
                              count      mean       std  min       25%       50%       75%  max
reward_history_rcos_0      107400.0  0.482964  0.269663  0.0  0.294203  0.494048  0.677090  1.0
reward_history_rcos_1      107400.0  0.452267  0.219151  0.0  0.292432  0.446573  0.606399  1.0
reward_history_rcos_2      107400.0  0.442919  0.228825  0.0  0.279929  0.442793  0.602768  1.0
reward_history_incomplete  107400.0  0.012961  0.113106  0.0  0.000000  0.000000  0.000000  1.0
```

All three features are finite (zero NaN) and inside [0, 1] as weighted reward
fractions; the indicator is strictly 0/1.

### Structural-missingness share

`reward_history_incomplete == 1` on 1,392 of 107,400 stops (**1.296 %**) and,
weighted by `n_selected_events`, on 1,039 of 65,913 events (**1.576 %**) —
matching the ≈1.3 % / ≈1.6 % expectation.

| animal | stops | flagged stops | % stops | events | flagged events | % events |
|---|---|---|---|---|---|---|
| chimi | 26,672 | 242 | 0.907 | 22,400 | 280 | 1.250 |
| j16 | 35,601 | 549 | 1.542 | 14,211 | 144 | 1.013 |
| peanut | 5,471 | 70 | 1.279 | 4,251 | 78 | 1.835 |
| senor | 19,611 | 221 | 1.127 | 17,890 | 404 | 2.258 |
| wilbur | 20,045 | 310 | 1.547 | 7,161 | 133 | 1.857 |
| **all** | **107,400** | **1,392** | **1.296** | **65,913** | **1,039** | **1.576** |

Every per-animal share falls inside the expected 0.4–2.5 % band on both
weightings.

### Prefix assertion

Within every one of the 218 (animal, session_date, epoch) groups, ordered by
`stop_idx`, the flagged stops form an **unbroken prefix** — asserted, not merely
inspected. All 218 sessions carry at least one flagged stop, as expected: every
session begins before two outcomes are known.

Consistency with the persisted lag columns: the flag fires on exactly the stops
where `known_reward_outcome_lag_2` is missing (1,392 of them; 833 of those still
have a known lag 1, the rest have none), and on no stop where two lags are
known. That is the "fewer than two known outcomes" definition, independently
recovered from the lag block.

## Scope of the change

- Modified: `Processed-Data/replay_content/stop_table.parquet` only.
- Not modified: every other canonical table, the models directories, and every
  prior backup directory. No file was deleted or moved.
- Downstream: fitted artifacts under `models_per_bin_rate/`, `models/`, etc.
  were produced against the pre-change stop table. Nothing in the 49 shared
  columns moved, so existing fits remain numerically valid; only ladders that
  actually reference the new basis columns need a refit.
