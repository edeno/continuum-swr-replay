# Event-patch-table regeneration for the allocation fraction

**Date:** 2026-09-05
**Code:** `173334a` (the commit that added the column)
**Artifact:** `Processed-Data/replay_content/event_patch_table.parquet`
**Backup of the pre-change table:** `Processed-Data/replay_content/_pre_fix_backup_20260905/event_patch_table.parquet`
**Candidate (staging):** `Processed-Data/replay_content/_staging_patch_fraction_full_20260905/event_patch_table.parquet`

## Decision

**Promoted.** The regenerated event-patch table adds one column
(`patch_fraction`) and changes nothing else: all 28 pre-existing columns are
identical to the backup in dtype, value, and NaN position, over the same
197,739 rows in the same `(event_id, patch)` order. The four other canonical
tables rebuilt on the way — and the standardization reference written beside
them — are byte-identical (SHA-256) to the live files and were not promoted.

`patch_fraction[e, p] = patch_count_hard[e, p] / (n_nonlocal_bins[e] −
n_undecoded_bins[e])`: the patch's share of the event's timebins whose MAP
estimate landed on a real arm. That is the same denominator
`build_event_arm_table` already uses for `arm_fraction`, so a patch fraction is
exactly the sum of its two leaf arms' and its trunk's `arm_fraction`, an event's
three patch fractions sum to one, and an event with no real-arm bin is NaN for
all three patches rather than 0/0. Only `n_nonlocal_bins` survives onto the
canonical arm table, so the builder carries both counts from the event table and
raises if either is absent.

**The covariate schema version stamp is unchanged at
`covariate_contract_v5_dio_poke_trials`.** Adding a column is not a vocabulary
change: the stamp records the behavioral-derivation contract (the DIO-poke trial
definition and the canonical names/row keys), and none of that moved. The
promoted table carries the identical `attrs` dict as the backup and passes
`require_covariate_schema(table="event_patch")`.

## Regeneration path and timing

`scripts/decoding/regenerate_stop_table.py` is not usable here (it calls
`load_data()`, which requires a live Spyglass/DataJoint connection). The rebuild
ran from the behavioral-context cache through the canonical driver, into a
staging directory — the same path the 2026-09-04 stop-table regeneration used:

```bash
tmux new-session -d -s patch_table_regen \
  "/usr/bin/time -v uv run python -m continuum_swr_replay.analysis.replay.prepare_data \
     --from-cache \
     --output-dir      Processed-Data/replay_content/_staging_patch_fraction_full_20260905 \
     --behavior-cache-dir Processed-Data/replay_content/_session_cache \
     --contingency-dir Processed-Data/replay_content \
     > logs/patch_fraction_regen/full.log 2>&1"
```

| run | scope | wall clock | peak RSS | exit |
|---|---|---|---|---|
| smoke (`--max-per-animal 1`) | 5 sessions | 3:45.24 | 11.74 GB | 0 |
| **full rebuild, tmux `patch_table_regen`** | **218 sessions** | **1:36:45** | **12.33 GB** | **0** |

Extrapolation from the smoke (45 s/session over 5 sessions) predicted ≈2.7 h for
218 sessions; the actual full run took 1 h 37 min (27 s/session), because the
smoke's per-animal-first sessions are among the larger ones — the same pattern
the stop-table regeneration saw. Peak RSS was 12.3 GB against 3.8 TB free, on a
96-core host with load average ≈0.5. No session was skipped and no error was
logged: the driver reported `events: 65913, event_arm rows: 593217, stops:
107400, sessions: 218` and the cross-table invariant
`Σ n_selected_events == len(event_table) == 65913`.

Logs: `logs/patch_fraction_regen/full.log` (smoke: `smoke.log`; parity report:
`parity.txt`).

Promotion, run only after every parity check below had passed. A plain copy, so
the promoted file is bit-for-bit the staged one and the staging directory
remains as its provenance:

```bash
cp -p Processed-Data/replay_content/_staging_patch_fraction_full_20260905/event_patch_table.parquet \
      Processed-Data/replay_content/event_patch_table.parquet
```

Verified afterward: the promoted file has the same SHA-256 as the staged one
and passes `require_covariate_schema(table="event_patch")`.

## Artifact integrity

| file | SHA-256 | rows |
|---|---|---|
| pre-change `event_patch_table.parquet` (= `_pre_fix_backup_20260905/`) | `ceb357984b2f1dfe326851a6dffa9f901405358c068d8a30c0e4a511760eff0d` | 197,739 |
| promoted `event_patch_table.parquet` | `dff5b2bb8312273da6a739dc69067a1e3aec455fdce358047b60ad6fa886c126` | 197,739 |

The pre-change digest is the same one recorded for `event_patch_table` in the
2026-09-04 stop-table audit and in `_golden_baseline_v5_v4cache/manifest.json`,
so the backup is the frozen golden baseline byte-for-byte.

The four other canonical tables were rebuilt by the same run and are
**byte-identical to the live files**; none was promoted (the live files were
never touched):

| table | SHA-256 (live == rebuilt) | rows |
|---|---|---|
| `event_table` | `68ffdc1bdce6881cd53a6f55ae813020907cd6eb86860a3d90a07a699869491c` | 65,913 |
| `event_arm_table` | `6b11ce36bcf23fa2b0d81cd56519ba6973a196e2f512ca1fd3011b7f2cca83f3` | 593,217 |
| `event_sequence_table` | `9f23c2b6071201c32434cf66492fe32112f43ef7f12f8006d84abcad73d7b8ed` | 2,971,614 |
| `stop_table` | `b261d9932163b34fdb604122f563fe9a42c1c7b88e5eb21b297b0999165b6b3e` | 107,400 |
| `standardization_reference.json` | `9a980d1faf342b799ef3de38f7a38c236c6baa55c31cd6098dd60f31157780ec` | — |

That the whole rest of the pipeline reproduces bit-for-bit is the strongest
available evidence that the rebuild environment matches the one that produced
the live tables, and that the only thing the new code changed is the patch
table. It also re-confirms, a day later and from a different process, the
determinism the stop-table audit established.

## Parity of the pre-existing columns

Checked by `scripts/qa/audit_event_patch_fraction.py`, which runs the shared
keyed-parity core (`table_parity.diff_table`, keys `animal`, `session_date`,
`epoch`, `event_id`, `patch`) with `patch_fraction` declared as an expected
addition, and adds the on-disk row-order and column-order checks the shared core
does not make (it sorts on the row key, so it is blind to the order rows and
columns are written in):

- rows: 197,739 vs 197,739
- `(event_id, patch)`: identical set **and** identical order
- schema attrs: identical dict (see the stamp note above)
- columns: 29 vs 28; the candidate-only set is exactly `patch_fraction`, and
  deleting it recovers the reference column order exactly (it is inserted after
  `patch_count_hard`, not appended)
- **28/28 pre-existing columns identical** in dtype, value, and NaN position

| column | dtype | dtype match | value mismatches | NaN-position mismatches | result |
|---|---|---|---|---|---|
| `event_id` | uint64 | yes | 0 | 0 | identical |
| `patch` | int64 | yes | 0 | 0 | identical |
| `patch_count_hard` | int64 | yes | 0 | 0 | identical |
| `is_current_patch` | float64 | yes | 0 | 0 | identical |
| `reward_count_candidate_arm_last10_known_trials_patch` | float64 | yes | 0 | 0 | identical |
| `reward_fraction_candidate_arm_last10_visits_patch` | float64 | yes | 0 | 0 | identical |
| `task_reward_probability_candidate_arm_patch` | float64 | yes | 0 | 0 | identical |
| `reward_belief_mean_candidate_arm_preoutcome_patch` | float64 | yes | 0 | 0 | identical |
| `visit_recency_trials_patch` | float64 | yes | 0 | 0 | identical |
| `is_patch_never_visited` | float64 | yes | 0 | 0 | identical |
| `is_last_rewarded_patch` | int64 | yes | 0 | 0 | identical |
| `is_last_rewarded_noncurrent_patch` | float64 | yes | 0 | 0 | identical |
| `is_prev_patch` | int64 | yes | 0 | 0 | identical |
| `is_swr` | bool | yes | 0 | 0 | identical |
| `swr_x_is_current_patch` | float64 | yes | 0 | 0 | identical |
| `swr_x_is_last_rewarded_noncurrent_patch` | float64 | yes | 0 | 0 | identical |
| `swr_x_is_prev_patch` | int64 | yes | 0 | 0 | identical |
| `animal` | object | yes | 0 | 0 | identical |
| `session_date` | int64 | yes | 0 | 0 | identical |
| `epoch` | object | yes | 0 | 0 | identical |
| `occupancy_state` | object | yes | 0 | 0 | identical |
| `at_well` | boolean | yes | 0 | 0 | identical |
| `occupancy_well_id` | float64 | yes | 0 | 0 | identical |
| `occupancy_well_arm` | float64 | yes | 0 | 0 | identical |
| `modal_observed_track_segment_id` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_min` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_mean` | float64 | yes | 0 | 0 | identical |
| `nearest_well_distance_cm_max` | float64 | yes | 0 | 0 | identical |

The per-column table above was produced at audit time; the script itself reports
the comparison in aggregate and raises on the first drifted column.

The comparison has teeth. Five mutants of the smoke-scale table were each
rejected: a `+1` on one `patch_count_hard` (int64), a 1e-6 perturbation of
`reward_fraction_candidate_arm_last10_visits_patch` (float64), a flipped
`is_swr` (bool), a 1e-6 perturbation of `patch_fraction` itself (caught by the
definition check, not by `diff_table`, since that column is a declared
addition), and a row reversal (caught by the order check, which `diff_table`
cannot see because it sorts on the key). Integer columns are compared exactly
rather than through a float cast: `event_id` is a uint64 hash, so a float
comparison would be blind to a one-unit change in almost every row.

## The added column

```
                 count      mean       std  min  25%  50%       75%  max
patch_fraction  197739.0  0.333333  0.435706  0.0  0.0  0.0  0.985294  1.0
```

Zero NaN over all 197,739 rows, every value inside [0, 1], and the mean is
exactly 1/3 — the arithmetic consequence of every event's three fractions
summing to one. The distribution is strongly bimodal: 104,538 rows (52.87 %) are
exactly 0 and 49,219 (24.89 %) are exactly 1, i.e. a quarter of all
event-patch rows are events whose decoded content sits entirely in one patch.

### Per-event sum check

Over all 65,913 events, `Σ_p patch_fraction = 1` with **max |sum − 1| =
2.220e-16** (one float64 rounding step; observed range
[0.99999999999999989, 1.0000000000000002]). Every event has at least one
real-arm bin, so **no event is all-NaN** in this cohort — the NaN branch is
exercised by the unit tests (`test_patch_fraction_nan_for_all_junction_event`)
rather than by the canonical data.

### Consistency with the arm table

`patch_fraction` equals the sum of `arm_fraction` over each patch's two leaf
arms plus its trunk, with **max |difference| = 2.220e-16** across all 197,739
rows. Since `arm_fraction` was not rebuilt (the arm table is byte-identical to
the live file), this is an independent check of the new column against a
quantity that predates it.

### Invoking the golden-baseline gate from now on

`table_parity.diff_table` refuses any undeclared column-set difference, so the
patch table now needs `patch_fraction` declared alongside the four stop-table
columns declared on 2026-09-04. **The baseline was not re-frozen.** The gate must
therefore be invoked with all five:

```bash
uv run python scripts/qa/compare_to_golden_baseline.py \
    --candidate Processed-Data/replay_content \
    --baseline  Processed-Data/replay_content/_golden_baseline_v5_v4cache \
    --tables event_table event_arm_table event_patch_table event_sequence_table stop_table \
    --expected-added-columns reward_history_rcos_0 reward_history_rcos_1 \
                             reward_history_rcos_2 reward_history_incomplete \
                             patch_fraction
```

Run against the promoted root this reports `matches baseline across 5 table(s),
ignoring declared added column(s)`. Dropping `patch_fraction` from the
declaration fails with `candidate-only=['patch_fraction']`, by design — verified
both ways at promotion time. Declaring a column a table does not carry is
harmless, so this one declaration covers the whole multi-table gate. The current
list is kept in the gate script's own docstring.

## Scope of the change

- Modified: `Processed-Data/replay_content/event_patch_table.parquet` only.
- Not modified: every other canonical table, the models directories, and every
  prior backup directory. No file was deleted or moved.
- Downstream: fitted artifacts under `models_patch/`, `models_patch_swr_*/` etc.
  were produced against the pre-change patch table. Nothing in the 28 shared
  columns moved, so existing fits remain numerically valid; only ladders that
  actually fit on `patch_fraction` need a new run.
