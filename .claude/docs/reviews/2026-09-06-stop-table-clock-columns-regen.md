# Stop-table regeneration for the two bout-start clock columns

**Date:** 2026-09-06
**Code:** `d7ef014` (the commit that added the columns)
**Artifact:** `Processed-Data/replay_content/stop_table.parquet`
**Backup of the pre-change table:** `Processed-Data/replay_content/_pre_phase3_backup_20260906/stop_table.parquet`
**Candidate (staging):** `Processed-Data/replay_content/_staging_stop_clocks_full_20260906/stop_table.parquet`

## Decision

**Promoted.** The regenerated stop table adds two columns
(`last_well_arrival_time_s`, `outcome_known_time_s`) and changes nothing else:
all 53 pre-existing columns are identical to the backup in dtype, value, and
NaN position, over the same 107,400 rows in the same `stop_id` order, across
the same 218 sessions.

`last_well_arrival_time_s` is the absolute time at which the animal arrived at
the well it is stopped at; `outcome_known_time_s` is the absolute time at which
that arrival's outcome became physically observable (pump onset when rewarded,
the canonical 0.30 s latency / poke-out boundary when omitted). Both are read
from the bout's *first* row (`strict_first`), so a bout that straddles an
arrival keeps the clocks of the trip it actually started under.

`outcome_known_time_s` is a **boundary**, not an outcome-gated value: it is
finite from the arrival onward, including at stops whose start precedes the
boundary. That is deliberate — a per-bin consumer has to know where the
boundary lies in order to split a stop at it — and it is why the column is
finite on 99.4 % of at-well stops while the outcome is actually *known* at only
80.1 % of at-well bout starts (see coverage below). The catalog nonetheless
classifies it `outcome_availability` / post-outcome, so the timing registry
keeps it out of any pre-outcome predictor gate.

**The covariate schema version stamp is unchanged at
`covariate_contract_v5_dio_poke_trials`.** Adding columns is not a vocabulary
change; the promoted table carries the identical `attrs` dict as the backup and
passes `require_covariate_schema(table="stop")`.

## Regeneration path and timing

`scripts/decoding/regenerate_stop_table.py` was not used, for the same reason
recorded in
[2026-09-04-stop-table-reward-basis-regen.md](2026-09-04-stop-table-reward-basis-regen.md):
it calls `load_data()` directly, which needs a live Spyglass/DataJoint
connection and would reload upstream behavior rather than reading the pinned
behavioral cache. The rebuild ran from the cache
(`_session_cache/comrie_photogate_trials_v4`, 218 entries) through the canonical
driver into a staging directory:

```bash
tmux new-session -d -s stop_table_regen \
  "/usr/bin/time -v uv run python -m continuum_swr_replay.analysis.replay.prepare_data \
     --from-cache \
     --output-dir      Processed-Data/replay_content/_staging_stop_clocks_full_20260906 \
     --behavior-cache-dir Processed-Data/replay_content/_session_cache \
     --contingency-dir Processed-Data/replay_content \
     > logs/stop_table_regen/full.log 2>&1"
```

| run | scope | wall clock | peak RSS | exit |
|---|---|---|---|---|
| smoke, `--stop-only` (`--max-per-animal 1`) | 5 sessions | 0:24.38 | 3.20 GB | 0 |
| smoke, full driver (`--max-per-animal 1`) | 5 sessions | 3:39.83 | 11.77 GB | 0 |
| **full rebuild, tmux `stop_table_regen`** | **218 sessions** | **1:34:36** | **12.46 GB** | **0** |

The full-driver smoke (44 s/session over its 5 sessions) extrapolated to ≈2.7 h
for 218 sessions; the actual run took 1 h 35 min (26 s/session), because the
per-animal-first sessions the smoke picks are among the larger ones — the same
direction and magnitude of over-estimate as the 2026-09-04 run. Peak RSS was
12.5 GB against ≈3.8 TB available. No session failed: the driver reported
`events: 65913, event_arm rows: 593217, stops: 107400, sessions: 218` and the
cross-table invariant `Σ n_selected_events == len(event_table) == 65913`.

Logs: `logs/stop_table_regen/full.log`.

Promotion, run only after every parity check below had passed — a plain copy, so
the promoted file is bit-for-bit the staged one and the staging directory
remains as its provenance:

```bash
cp -p Processed-Data/replay_content/_staging_stop_clocks_full_20260906/stop_table.parquet \
      Processed-Data/replay_content/stop_table.parquet
```

Verified afterwards: the promoted file has the same SHA-256 as the staged one
and passes `require_covariate_schema(table="stop")`.

## Artifact integrity

| file | SHA-256 | rows |
|---|---|---|
| pre-change `stop_table.parquet` (= `_pre_phase3_backup_20260906/`) | `b261d9932163b34fdb604122f563fe9a42c1c7b88e5eb21b297b0999165b6b3e` | 107,400 |
| promoted `stop_table.parquet` | `ea62e0b65fbec97c8fce64185a6627856a7056b1f5e3915aef52bf32d944610f` | 107,400 |

The pre-change digest is the one recorded as promoted by the 2026-09-04
regeneration, so the live table had not moved since.

The other four canonical tables were rebuilt by the same run and are
**byte-identical (SHA-256) to the live files**; none was promoted (the live
files were never touched):

| table | SHA-256 (live == rebuilt) | rows |
|---|---|---|
| `event_table` | `68ffdc1bdce6881cd53a6f55ae813020907cd6eb86860a3d90a07a699869491c` | 65,913 |
| `event_arm_table` | `6b11ce36bcf23fa2b0d81cd56519ba6973a196e2f512ca1fd3011b7f2cca83f3` | 593,217 |
| `event_patch_table` | `dff5b2bb8312273da6a739dc69067a1e3aec455fdce358047b60ad6fa886c126` | 197,739 |
| `event_sequence_table` | `9f23c2b6071201c32434cf66492fe32112f43ef7f12f8006d84abcad73d7b8ed` | 2,971,614 |

That the rest of the pipeline reproduces bit-for-bit is the evidence that the
rebuild environment matches the one that produced the live tables and that the
only thing the new code changed is the stop table. (The `event_patch_table`
digest differs from `_golden_baseline_v5_v4cache`'s because that table gained
`patch_fraction` on 2026-09-05; it matches the *live* file exactly.)

## Parity of the pre-existing columns

Checked with the shared keyed-parity core (`table_parity.diff_table`, `atol=0`,
key `stop_id`) declaring the two new columns as expected additions, plus the
on-disk row-order and column-order checks the shared core does not make:

- rows: 107,400 vs 107,400; sessions: 218
- `stop_id`: identical set **and** identical order (`Series.equals`)
- schema attrs: identical dict
- columns: 55 vs 53; the candidate-only set is exactly the two added columns,
  and deleting them recovers the reference column order exactly
- dtypes: no mismatch on any of the 53 shared columns
- **53/53 pre-existing columns identical** in value and NaN position — the
  `diff_table` drift report is empty

The comparison has teeth for the same reasons recorded on 2026-09-04: integer,
boolean, and object columns are compared exactly rather than through a float
cast, so a one-unit change in a `uint64` `stop_id` (107,301 of which exceed
2**53) or in a nullable `Int64` column is caught.

Two independent builder entry points agree: on the 5-session smoke scope, the
`--stop-only` build and the full-driver build produce identical values on all 53
shared columns (`diff_table` drift report empty); the two columns `--stop-only`
omits are the assembly-level `n_nonlocal_swr_events` /
`n_nonlocal_non_swr_events` split.

### Golden-baseline gate

The two new names join the running declaration list (the gate script's docstring
holds the current list). Against the frozen `_golden_baseline_v5_v4cache`:

```bash
uv run python scripts/qa/compare_to_golden_baseline.py \
    --candidate Processed-Data/replay_content \
    --baseline  Processed-Data/replay_content/_golden_baseline_v5_v4cache \
    --tables event_table event_arm_table event_patch_table event_sequence_table stop_table \
    --expected-added-columns reward_history_rcos_0 reward_history_rcos_1 \
        reward_history_rcos_2 reward_history_incomplete patch_fraction \
        last_well_arrival_time_s outcome_known_time_s
```

reports `matches baseline across 5 table(s), ignoring declared added column(s)`.
(The baseline root holds only those five tables, so `--tables` must be passed;
the default table list includes `qvals_table`, which the baseline predates.)

## The added columns

Both are absolute session times in seconds (unix-epoch clock, so their means are
≈1.6e9 and only differences are interpretable).

### Coverage by occupancy state

`arrival_finite` / `outcome_finite` are the share of stops where the column is
finite; `outcome_known_at_bout_start` is the share where the outcome had
*already* become observable by the bout start (`stop_start >= outcome_known_time_s`)
— the quantity a pre-outcome gate would use.

| occupancy state | stops | arrival finite | outcome finite | outcome known at bout start |
|---|---|---|---|---|
| `at_well` | 69,224 | 99.40 % | 99.40 % | 80.06 % |
| `in_transit` | 22,850 | 98.56 % | 98.56 % | 98.56 % |
| `mixed` | 14,845 | 99.31 % | 99.31 % | 85.16 % |
| `unknown` | 481 | 0 % | 0 % | 0 % |
| **all** | **107,400** | **98.77 %** | **98.77 %** | **84.34 %** |

The 80.1 % figure for at-well stops matches the expectation stated when the
columns were specified. Its complement is the stop that begins in the gap
between arrival and feedback — exactly the population the per-bin
outcome-boundary split exists to handle. In-transit stops are essentially all
past the boundary, as expected: they begin after the animal has left the well.

Per animal, at-well stops only:

| animal | at-well stops | arrival finite | outcome known at bout start |
|---|---|---|---|
| chimi | 19,239 | 99.52 % | 80.61 % |
| j16 | 20,065 | 99.31 % | 90.86 % |
| peanut | 4,094 | 99.41 % | 73.62 % |
| senor | 15,037 | 99.53 % | 67.17 % |
| wilbur | 10,789 | 99.18 % | 79.38 % |

### Missingness is exactly the pre-first-arrival prefix

Both columns are missing on the same 1,325 stops (0 % NaN-pattern disagreement
between them), and:

- missingness coincides exactly with a missing `completed_trial_id` — the
  independent "no completed trip yet" marker — on all 107,400 rows;
- within every one of the 218 sessions, ordered by `stop_idx`, the missing block
  is an unbroken **prefix** (asserted, not inspected), and every session has at
  least one such stop, as expected: a session begins before its first arrival;
- by state the 1,325 split as 414 `at_well`, 328 `in_transit`, 102 `mixed`,
  481 `unknown`.

### Ordering and magnitude sanity

- `last_well_arrival_time_s <= stop_start` on **every** finite row (minimum
  slack 0.26 ms) — the arrival never postdates the stop it anchors.
- `outcome_known_time_s - last_well_arrival_time_s` ∈ [0.0223 s, 0.300 s],
  median 0.062 s. The upper end is the omission latency cap
  (`OMISSION_KNOWN_LATENCY_SEC = 0.30`), which 47.3 % of stops sit exactly at;
  the rest are pump onsets, i.e. shorter and continuously distributed. The split
  is consistent with the task's reward rate.
- `stop_start - last_well_arrival_time_s` (dwell before the stop) has median
  0.79 s and a long right tail (max 1,331 s), as expected for bouts that begin
  long after an arrival.

## Scope of the change

- Modified: `Processed-Data/replay_content/stop_table.parquet` only.
- Not modified: every other canonical table, the models directories, and every
  prior backup directory. No file was deleted or moved.
- Downstream: fitted artifacts under `models_per_bin_rate/`, `models/`, etc.
  were produced against the pre-change stop table. Nothing in the 53 shared
  columns moved, so existing fits remain numerically valid; only designs that
  reference the new clocks need anything rebuilt.
