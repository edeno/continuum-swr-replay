# Phase 3a (C, code) — Marked family v2 for content, strength and context

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: questions 3, 5 and 8 are graded properties of events (what is replayed, how strong the event is) and of the stop (reward outcome, how close a patch switch is). The marked family estimates how the peri-event kernel changes per within-session SD of such a property. Its first corpus showed that context and content "effects" were partly stationary rate differences between stops; the v2 plan's per-stop baseline terms, marginal tier and shuffled-mark reference are the fixes. This phase implements the parts of v2 the three questions need, plus three models that ask them. Code only.

**Inputs to read first** (after phase 0, on this branch):

- `.claude/docs/plans/pfc-marked-family-v2/phase-1a-baseline-terms.md` — baseline columns: the builder code, `KernelDesign.n_baseline`, penalties, contract 2. **Implement it as written**, with the extension below.
- `.claude/docs/plans/pfc-marked-family-v2/phase-2-marginal-tier.md` and `phase-3-slope-permutation.md` — the marginal tier and the shuffled-mark reference. Implement as written, generated from the new models below instead of the v1 core list.
- `.claude/docs/plans/pfc-marked-family-v2/shared-contracts.md` — baseline columns, contract 2 keys.
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py` — `MARK_DESCRIPTIONS`, `MarkedFamilySpec`, `RECOMMENDED_MODEL_SPECS`, `build_recommended_event_mark_table`, `build_marked_family_ladder`, `fit_marked_family_ladder` (re-grep the line numbers after phase 0).
- `Processed-Data/replay_content/stop_table.parquet` (`current_patch`, `next_patch`, `stop_idx`, `stop_start`, `is_rewarded_outcome`) and `event_table.parquet` (`ripple_power_z`, `mua_rate_z`, `event_duration_s`).

**Contracts referenced:**

- v2 [Baseline columns](../pfc-marked-family-v2/shared-contracts.md#baseline-columns) and [Fit contract version 2](../pfc-marked-family-v2/shared-contracts.md#fit-contract-version-2); [Bundle trees](shared-contracts.md#bundle-trees).

## Tasks

- **Baseline terms** — v2 phase 1a in full. Extension: `build_baseline_columns(..., context_columns: Sequence[str] = ())` adds one unpenalised per-stop column for each stop-level context mark a model targets (`trips_to_patch_switch` as its stop value, z-scored over fitted bins, with an `_unknown` indicator where undefined). Contract key `baseline_context_columns`.
- **New mark `trips_to_patch_switch`** in `build_recommended_event_mark_table`: per session, order stops by `stop_idx`; for stop `s`, the number of completed trips from `s` until the first later trip whose destination patch differs from its origin (0 when the next trip leaves the patch), capped at 3; NaN after the session's last switch. Derive from the stop table's `current_patch`/`next_patch` (verify on one session against the trials table that `next_patch` is the destination of the trip that starts at this stop). Plus `next_trip_switches_patch = (trips_to_patch_switch == 0)`. Add both to `MARK_DESCRIPTIONS`. Question 8's definition is the lead's (overview Open Questions 7); this is the default.
- **Three models** in a new mapping `FOLLOWUP_MODEL_SPECS` (the v1 `RECOMMENDED_MODEL_SPECS` stays; nothing is deleted):

  | model | targets | adjustments | baseline context |
  |---|---|---|---|
  | `replay_content` | `current_arm_fraction`, `content_weighted_recent_reward`, `content_weighted_belief_uncertainty` | `log_event_duration_s`, `leaf_content_fraction`, `posterior_entropy_mean` | — |
  | `event_strength` | `log_event_duration_s`, `ripple_power_z`, `mua_rate_z` | `posterior_entropy_mean` | — |
  | `stop_context` | `rewarded_context`, `trips_to_patch_switch` | `log_event_duration_s`, `current_arm_fraction`, `posterior_entropy_mean` | `trips_to_patch_switch` |

  Both replay classes are marked (`MARKED_EVENT_CLASSES`). Reward outcome is already a baseline column under v2 1a.
- **Marginal tier** (v2 phase 2) generated from `FOLLOWUP_MODEL_SPECS`: each target alone with `log_event_duration_s` and `posterior_entropy_mean` as adjustments (`log_event_duration_s` alone with entropy), content targets also with `leaf_content_fraction`; `next_trip_switches_patch` gets a marginal spec too.
- **Shuffled-mark reference** (v2 phase 3) for the three joint models and every marginal spec, default R = 30, permuting each target within class among events where it is defined; for stop-level marks (`rewarded_context`, `trips_to_patch_switch`) permute **whole stops' values among stops** (all events of a stop move together), so the reference keeps the within-stop dependence. Write `.slope_null.npy` companions.
- **Runner** `run_pfc_marked_model_family.py`: `--model-set {recommended,followup}` (default `recommended`), `--tier`, `--slope-permutations`, output default unchanged; the follow-up runs pass `--output-dir Processed-Data/pfc_response_followups/marked_family`. Contract 2.
- **Summary** `summarize_pfc_marked_family.py`: refuse mixed contracts; per model × class × target the population mean slope kernel (unit → session → animal), the null band, fraction of units gaining held-out deviance, joint beside marginal. Writes `Processed-Data/pfc_response_followups/marked_family/summary.parquet` (one row per model, tier, class, target, statistic) for phase 4 to read.
- **Docs**: `docs/design/pfc_marked_model_family.md` — baseline terms, the follow-up models table, the stop-level permutation rule, the new mark; `scripts/README.md` flags.

## Deliberately not in this phase

- No GPU run (phase 3b).
- v2 phase 1b's re-specification of the v1 models, its interaction work, and v2 phases 4–5.
- No marks on `SWR_no_non_local` (overview Open Questions 8).

## Validation slice

| Test | Asserts |
| --- | --- |
| v2 phase 1a's ten tests | as listed there |
| `test_trips_to_patch_switch_counts_forward` | a synthetic stop sequence with patches 1,1,1,2,2,3: values 2,1,0,1,0,NaN; cap at 3 |
| `test_context_baseline_column_is_constant_within_stop` | the `trips_to_patch_switch` baseline column is constant on each stop's fitted bins and unpenalised |
| `test_followup_specs_use_known_marks` | every target/adjustment is in `MARK_DESCRIPTIONS`; widths per class recorded |
| v2 phase 2's marginal tests | on the follow-up specs |
| `test_stop_level_permutation_moves_whole_stops` | after permutation every stop's events share one value and the multiset of stop values is unchanged |
| `test_planted_context_kernel_is_separated_from_its_baseline` (`slow`) | generator with a ×1.5 stationary rate on rewarded stops and no event-locked difference: with baseline, the `rewarded_context` slope kernel's mean over ±200 ms lies inside its null band; without baseline it does not |
| `test_planted_strength_kernel_is_recovered` (`slow`) | a planted per-SD gain on `ripple_power_z` is recovered (sign and peak lag) and lies outside its null band |

## Fixtures

The v1 test generators in `tests/analysis/pfc/test_marked_model_family.py`, extended with stops and a context offset.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
