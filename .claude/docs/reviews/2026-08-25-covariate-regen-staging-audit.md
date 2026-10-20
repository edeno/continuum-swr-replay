# Covariate regeneration staging audit

**Date:** 2026-08-25  
**Code:** `766ea7d99b4fb1eb28771808a21a3b008d6207df` (`collapse-covariate-boundary`)  
**Candidate roots:** `Processed-Data/replay_content_regen_staging/` and
`Processed-Data/contfrag_replay_content_regen_staging/`  
**Baseline:** `/tmp/continuum-covariate-rename-baseline-20260822/` and
`Processed-Data/replay_content/_pre_canonical_regen_backup_20260824_142611/`

## Decision

**Do not promote the staging directories yet.** The regenerated Replay and ContFrag
base tables pass their scientific row contracts, and the numerical changes are
traceable. The staged bundle is nevertheless incomplete because its q-values table
predates three recovered sessions and lacks its required standardization reference.
Final model inference is additionally blocked by rank-deficient post-outcome designs
and by unequal row identities in the current rate-model ladders.

This audit was read-only. No candidate, paper-facing, or backup artifact was modified.
All eight current paper-facing Replay parquet files are byte-identical to their dated
pre-regeneration backups, including `qvals_table.parquet`.

## Artifact integrity and rename parity

- Every regenerated table carries the `covariate_names_v1` schema stamp.
- Replay and ContFrag event-arm tables have exactly nine candidate arms per event;
  patch tables have exactly three patches per event.
- Event-ID/session sets, response conservation, fraction normalization, uniform leaf
  support, unique keys, and schema contracts all pass on the materialized candidates.
- Replay event, stop, and patch values/dtypes/missingness are exact on the 215-session
  baseline intersection. The candidate adds three sessions and drops none.
- The q-values table is exactly identical to the pre-rename table after table-scoped
  canonicalization: 103,852 row keys and 65 non-key columns match exactly.
- ContFrag differences are confined to the three sessions whose trial caches were
  refreshed after the checksum failure. Recomputing from the archived and refreshed
  caches reproduces the baseline and candidate values exactly. The only independent
  schema addition is `arm_fraction_posterior`, which equals
  `arm_posterior_mass / n_timebins` exactly and sums to one within every event.

The three recovered sessions are:

| animal | session_date | epoch | added events | added stops |
|---|---:|---|---:|---:|
| j16 | 20210707 | 12_r6 | included in the 584-event total | 780 |
| senor | 20201102 | 02_r1 | included in the 584-event total | 298 |
| wilbur | 20210331 | 12_r6 | included in the 584-event total | 453 |

Their explicit cache parity exceptions are bound by these SHA-256 pairs:

| session | archived cache | refreshed cache |
|---|---|---|
| j16 20210707 12_r6 | `4fdaf0961bdaadb859a2f3083385e8ed752ceaa334517d7013dd61a79a057eb6` | `18977f1870e1593e577f19a11eb82d942a5ddc0a83a30a3d157daac26a9754bf` |
| senor 20201102 02_r1 | `200b186cb03a13926b9fd6384f2a6d193023fcaf83779b2a75870c6173998617` | `a0c186a634f41c709616423f870d7072a0266a85b1cd658397514784181975b4` |
| wilbur 20210331 12_r6 | `61ec346abc553a68709767ea83fed6abd591fbe88fa639fd7b62f670f95655e0` | `8606ab5a769253ea9bf904bcf8173278442892a019374380429d5351e33f4b43` |

## Candidate inventory

| artifact | rows | columns | sessions | SHA-256 |
|---|---:|---:|---:|---|
| Replay `event_table.parquet` | 65,913 | 103 | 218 | `b35145d0e69442b437062f692ca3178ee7bda366095064151f9522871e408616` |
| Replay `event_arm_table.parquet` | 593,217 | 84 | 218 | `a59e86c3a81a8e8f5cf4f8d7fdbb70f2943ec4b8ef6e1628ce3fd40a83c85651` |
| Replay `event_patch_table.parquet` | 197,739 | 28 | 218 | `ab414a2f2674cb34e7fdcfb500ef655a87739b0102a0af47f6f1d0bd8c32e0b8` |
| Replay `event_sequence_table.parquet` | 2,971,614 | 18 | 218 | `a15042c5abc36cbee968bff708fd30c5e7d2578edc23d2b12938791abca0f28c` |
| Replay `stop_table.parquet` | 107,400 | 49 | 218 | `9acb5eb583c421beaf3f330d8b217ae493f0fc977578dc8683b90e9334c30139` |
| Replay `event_onset_table.parquet` | 137,745 | 11 | 218 | `23e2e1c397fd2c8fb2184aa15f2b37c9a783c0a3938a5e7f5755533e051ae344` |
| Replay `stop_timebin_table.parquet` | 32,533,075 | 12 | 218 | `a646d92197b4117480dd4518ce4fd23a7e4d171889e6d363ff06be5aa77b8ee8` |
| Replay `qvals_table.parquet` | 103,852 | 66 | 215 via stop-ID mapping | `039ec238731430c2983e8215ce9ca039d8d3259b10664561ff2770ba11f3a83b` |
| ContFrag SWR `event_table` | 38,558 | 80 | 218 | `aca394071dd693335475fbac7912b47044da521f34970fa5960764def8f169eb` |
| ContFrag SWR `event_arm_table` | 347,022 | 88 | 218 | `dca7d4763e03ca2982abbee470c4d2611939c600ac77f283a77e53e55a37753a` |
| ContFrag SWR `event_patch_table` | 115,674 | 30 | 218 | `7694f648f8eac0f2234727d5a42d4be21045d0be1304815f57a1d08ffc8e1e49` |
| ContFrag SWR `stop_table` | 107,400 | 36 | 218 | `ffbe3620c4c431e8cf8d687e350c7fa5e342ce223e05d694bb8eaecab4161e0b` |
| ContFrag MUA `event_table` | 33,255 | 80 | 218 | `f3f08476a8ab6cf1af8884e4e7312560499763be09f822a3efd21495711d9df0` |
| ContFrag MUA `event_arm_table` | 299,295 | 88 | 218 | `35a26cb9351cdbe29a0de3a6a051c2745aeb410408b6ed0e2a6958e1620b2550` |
| ContFrag MUA `event_patch_table` | 99,765 | 30 | 218 | `ff552669808777c9e169591780b4493d4f1a65f5f3f3ee878f7199ffd38a238a` |
| ContFrag MUA `stop_table` | 107,400 | 36 | 218 | `c71e02dac865fae7ed8cfa9b0d5e7bf553d5007379e790c9e4af6500b2f70ed6` |

The event-arm standardization reference is present and internally exact: reference
row counts equal finite leaf-arm support, leaf-support z-scores have mean zero and
population SD one, and applying the persisted moments reproduces every `_z` value
exactly. The q-values table contains six `_z` columns but its required sibling
`models_per_bin_rate/qvals_standardization_reference.json` is missing.

## Cohort retention

### Replay content

- 65,913 events enter; 59,918 (90.9%) are wholly at one well. All 218 sessions
  retain at least one at-well event.
- The default q-independent arm/patch union loses three early-history events and no
  sessions: 59,915 events across all 218 sessions. M0-M6 and P0-P3 pass the
  pre-fit sample/rank gates for every session.
- Belief/value rungs retain 58,907 events. The 1,011-event complete-case loss is
  the 1,008 at-well q-values gaps plus the three early-history events.
- Arm Mv/Mu/Mvu retain 215 sessions before numerical fitting. Patch Pv retains 213:
  it additionally loses `senor 20201027 02_r1` and `j16 20210706 02_r1` to
  within-session rank deficiency. A matched P3/Pv comparison therefore retains
  57,890 events across 213 sessions.

The q-values coverage gap is not an internal alignment failure:

- 103,852 / 107,400 stops (96.696%) map exactly and uniquely.
- The 3,548 unmapped stops partition into 1,531 stops from the three recovered
  sessions and 2,017 post-final-trial suffix stops across 208 sessions.
- There are zero leading or internal stop-order holes among the 215 q-covered
  sessions; seven sessions are fully covered.
- The gap affects 1,465 Replay events: 584 in the three whole missing sessions and
  881 in terminal tails.

The whole-session gap is an artifact-order loss: the staged q-values table was
written before the recovered 218-session stop table. It should not be accepted as
the scientific cohort definition.

### ContFrag content

- SWR: 38,558 raw -> 36,532 at-well -> 36,530 shared complete-case events.
  Patch P0-P2 and arm M0-M3 retain all 218 sessions. Arm M4 retains 35,997 events
  across 214 sessions; four sessions have an all-zero novelty column and therefore
  a rank-deficient M4 design: `chimi 20200217 02_r1`,
  `chimi 20200224 02_r1`, `senor 20201028 02_r1`, and
  `senor 20201104 02_r1`.
- MUA: 33,255 raw -> 31,256 at-well/complete-case events. Patch P0-P2 retain all
  218 sessions. Arm retention is M0 218, M0a 217, M1 216, M2/M3 215, and M4 209
  sessions (30,605 events), from a combination of small samples and an all-zero
  novelty column. The final M4/shared-ladder losses are
  `chimi 20200218 06_r3`, `senor 20201028 02_r1`,
  `senor 20201030 02_r1`, `senor 20201101 14_r7`,
  `senor 20201103 10_r5`, `senor 20201103 14_r7`,
  `senor 20201104 02_r1`, `senor 20201107 12_r6`, and
  `wilbur 20210331 02_r1`.

The current content runner intersects successful sessions across every requested
rung. Consequently an all-rung SWR run would reduce otherwise estimable M0-M3 fits
to 214 sessions, and an all-rung MUA run would reduce M0 to 209. That is exact-data
matching, but it propagates an optional complex-rung failure farther than necessary.
Formal comparisons should instead freeze named parent-child comparison sets; simpler
estimation outputs should retain their own maximal estimable cohorts.

### Per-stop and per-bin rate ladders

Every session passes preliminary size/variation gates, but the current runners fit
each rung on its own complete cases:

| predictor support | stops | 15-ms bins |
|---|---:|---:|
| duration only (R0; M0/M3/M5/M6/M3r) | 107,380 | 7,400,248 |
| lag-1 / recent fraction (R1/R2; M1/M2r) | 106,457 | 7,353,551 |
| full ten-outcome history (Rlag; M2/M4/M7/M9/M7r) | 100,307 | 6,857,693 |

Thus current nested comparisons do not use the same observations. Examples:

- M0 -> M1 loses 923 parent stops.
- M1 -> M2 loses a further 6,150 stops.
- M3 -> M4 and M6 -> M7 lose 7,073 parent stops.
- R0 -> Rlag loses 7,073 parent stops.

AIC/LRT/CV comparisons across those pairs are invalid until each parent is refit on
the child's exact stop/bin keys (or both are fit on an explicitly named comparison
set). A single all-ladder union is unnecessary and would discard more rows than each
question requires.

### Post-outcome models

SWR power:

- 65,913 events -> 64,448 with q-values/timing -> 59,680 physically post-outcome ->
  59,327 complete cases.
- The outcome-balance guard retains 59,142 events in 214 sessions. Three sessions
  are lost only because of the stale q-values bundle; `chimi 20200228 02_r1` has no
  retained omission and is the one scientifically balance-ineligible session.
- 142 / 214 selected session designs are rank deficient. In 113 sessions,
  `rpe_postoutcome + reward_belief_mean_chosen_arm_preoutcome` is exactly equal to
  `known_reward_outcome_lag_1`, while all three enter the same OLS. The current
  `np.linalg.lstsq` fit silently returns a minimum-norm solution without a rank guard.

Uncertainty rate:

- 218 sessions -> 215 after the stale q-values join -> 210 after the intended first
  exposure-session removal.
- 6,952,425 bins -> 6,409,208 physically post-outcome -> 5,848,940 complete cases.
  All 210 sessions retain rewarded and omitted bins for all three fitted populations.
- Rewarded-bin retention is 90.1% from input; omission retention is 62.5%. This
  asymmetry must be reported with the model result.
- `task_reward_probability_patch_mean_gap_best_vs_second` is constant within 41
  session designs, making 123 session x population GLMs rank deficient. The generic
  per-bin fitter checks lag-1 variation but not full matrix rank before statsmodels.

## Code review

### Strong aspects

- The `covariates/` package now separates clocks, behavioral context, history,
  candidate-arm assembly, and definitions; the previous `_helpers` ownership cycle
  is removed while public imports remain stable.
- Replay and ContFrag use the same candidate-arm derivation and scientific table
  contracts. Exact alignment, candidate sets, response conservation, structural
  support, and posterior validity fail loudly.
- Arrival and outcome clocks are distinct, stable-sort invariant, and explicitly
  tested. Q-values preserve authoritative timing intervals around missing model rows
  and verify reward/RPE identities.
- The three cache-driven numerical changes were reproduced in both directions, not
  accepted merely because they looked plausible.

### Required fixes

1. **Make staging an explicit supported workflow.** `prepare_data.py`,
   `qvals_table.py`, and the ContFrag prep script hard-code paper-facing input/output
   directories. Add explicit `--input-dir` / `--output-dir` arguments, refuse identical
   source/destination for audit mode, and publish a complete directory atomically.
   This is required to regenerate q-values against the candidate stop table without
   overwriting the old artifact.
2. **Regenerate q-values after all 218 stop sessions exist.** The timing/model outer
   audit currently fetches timing only for NWBs already present in model values, so a
   wholly absent model session is invisible. Use the expected stop/timing session set
   as the authority, persist per-session gap reasons, and require the standardization
   sidecar in the same bundle.
3. **Repair the machine-readable catalog.** It currently describes patch recency as a
   leaf mean and patch novelty as “at least one novel leaf”; implementation, tests, and
   the human dictionary use the leaf minimum and require both leaves novel. It also
   says `completed_trial_reward` is missing before feedback even though that explicitly
   descriptive/future-bearing column stores the eventual outcome after arrival.
4. **Add full-rank guards and reformulate post-outcome designs.** Remove one member of
   the exact lag1/RPE/chosen-belief identity in the SWR-power model. Remove or redesign
   the session-constant patch-gap term for per-session uncertainty fits. Never accept
   pseudo-inverse coefficients as identified estimates.
5. **Freeze comparisons by declared edge.** Rate parents must be refit on exact child
   stop/bin keys. Content selection should use named comparison sets rather than a
   single all-requested-rung successful-session intersection.
6. **Fix audit attribution.** `QVALS_GAP_COLS` omits the standardized candidate-arm
   variance and patch-level q-values aggregates, so Mu/Pv would currently call their
   1,008 at-well q-values-gap events generic own-predictor missingness. The observed
   incorrect audit is 0 q-values gaps + 1,011 own-predictor losses; the correct split
   is 1,008 q-values gaps + three early reward-history losses.
7. **Persist a release manifest.** The schema stamp intentionally certifies only names
   and row keys. Record git/lock/config/constants, expected/succeeded/skipped session
   keys and reasons, input/cache hashes, every output hash/shape/schema/key-set hash,
   standardization references, contract/parity results, and the atomic promotion /
   rollback transaction. DataFrame attrs are insufficient and q-values per-animal
   attrs are lost by concatenation.
8. **Tighten small maintainability gaps.** The definition/history module remains large;
   `history.py` mixes derivation, local-Q utilities, and artifact attachment. Patch
   contracts should explicitly conserve response totals, event-arm totals used for
   conservation should be required constant across candidate rows, and the current
   `table_contracts.py` import block should pass Ruff. ContFrag `arm_count_soft` is
   exactly identical to `arm_count_hard` on all 646,317 staged candidate rows because
   the ContFrag builder uses unit weights; use posterior mass for a real robustness
   response or omit the vacuous duplicate. Update
   `docs/known_issues/three_sessions_blocked_by_nwb_checksum.md`: the three sessions
   are recovered in Replay/ContFrag, while q-values alone remain at 215.

## Verification performed

- 324 focused tests passed across definitions, table contracts, q-values, cohort
  selection/auditing, content runner validation, per-stop/per-bin models, trajectory
  parity, and posterior summaries.
- A second covariate/readability slice passed 251 tests.
- Materialized Replay and ContFrag table contracts passed directly.
- Exact baseline/candidate checks passed on Replay event, patch, stop, and q-values;
  the three-session ContFrag exceptions were reproduced from archived/refreshed caches.
- The scoped Ruff run found one import-order error in `table_contracts.py`; this audit
  did not modify production code.

Promotion can be reconsidered after the q-values candidate and its reference are
rebuilt in staging, the manifest is complete, the definition catalog is corrected,
and the rank/cohort comparison blockers are resolved or explicitly excluded from the
promotion's claim scope.
