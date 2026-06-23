# Phase 2 — Trajectory rollups in both event tables

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Add the per-event trajectory-metric columns to **both** `build_event_table`
(Non-Local) and `build_contfrag_event_table` (ContFrag) from one shared module,
and merge the already-existing `path_continuity` and retrospective metrics
(plus a new `prev_arm` mirror) into the canonical tables. Depends on Phase 1's
Tier-1 columns.

**Inputs to read first:**

- [tables.py:313-533](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L313-L533) — `build_event_table` per-event loop and section-7 rollup-assignment pattern (485-527). New rollups are assigned here.
- [tables.py:1351-1418](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1351-L1418) — `_event_table_columns`: append new columns here.
- [tables.py:613-712](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L613-L712) — `build_event_sequence_table`; [tables.py:1534-1548](../../../../src/continuum_swr_replay/analysis/replay/tables.py#L1534-L1548) — its column list. Extend only if a rollup needs a per-bin column not already carried.
- [uncertainty.py:28-204](../../../../src/continuum_swr_replay/analysis/replay/uncertainty.py#L28-L204) — the `_weighted_mean_by_event` / `(unique_ids, array)` pattern to mirror.
- [contfrag/tables.py:163-449](../../../../src/continuum_swr_replay/analysis/contfrag/tables.py#L163-L449) — `_summarize_event_state` (163-207) and `build_contfrag_event_table` per-event loop + posterior section; [contfrag/tables.py:862-909](../../../../src/continuum_swr_replay/analysis/contfrag/tables.py#L862-L909) — its column list.
- [spatial.py:120-192](../../../../src/continuum_swr_replay/analysis/replay/spatial.py#L120-L192) — `arm_adjacency` + `path_continuity`.
- [retrospective.py:79-468](../../../../src/continuum_swr_replay/analysis/replay/retrospective.py#L79-L468) — `_PATCH_TO_TRUNK_ARM` (79), the path helpers, and `compute_retrospective_metrics` (429-468).
- [constants.py:152-178](../../../../src/continuum_swr_replay/data_loaders/constants.py#L152-L178) — `WELL_PATCH_MAPPING`, `TRACK_SEGMENT_TO_PATCH`.

**Designs referenced:** [designs.md](designs.md) — the full `trajectory.py` module contract, helper code, and per-column definitions.

## Tasks

- **Create `analysis/replay/trajectory.py`** implementing the four rollup
  families and helpers in [designs.md](designs.md): `trajectory_dynamics`,
  `brain_state_means`, `ahead_behind_summary`, `uncertainty_extra`, plus
  `_linear_to_node` / `_graph_dist`. Each returns `(unique_ids, {col: array})`
  exactly like `uncertainty.py`. Weighting is an explicit argument.
- **Wire rollups into `build_event_table`** in the section-7 block
  (`tables.py:485-527`): build the all-pairs `dist_lookup` once, call each
  family with `weights=nonlocal_prob_arr`, and assign every returned column.
  Pass `env`, `run_speed_baseline` (read from the in-event `dot`), and
  `place_bin_size`.
- **Merge `path_continuity`**: compute per event over the in-event
  `decoded_arm` sequence using `arm_adjacency(track_graph)`; assign column
  `path_continuity`.
- **Add the `prev_arm` retrospective mirror** in `retrospective.py`
  (`ends_at_prev`, `contains_path_from_prev`, `chosen_arm_fraction_prev`) and
  have `compute_retrospective_metrics` emit them alongside the prospective set.
  Reuse the existing path helpers with `prev_*` substituted for `next_*`.
- **Wire `compute_retrospective_metrics` into the event-table build** (or its
  immediate caller / Stage 5B assembly) so the prospective + retrospective
  columns land in the canonical `event_table`, merged on `event_id`. Name the
  current standalone call site that this replaces and remove the redundant pass.
- **Apply the same rollups to `build_contfrag_event_table`** with
  `weights=np.ones(...)`. Replace the bespoke `_summarize_event_state` pair
  (`continuous_prob_mean`/`fragmented_prob_mean`) with `brain_state_means`
  (which produces all four state means identically); keep the ContFrag-only
  `dominant_state` / `continuous_fraction` as-is.
- **Update both column-list helpers** (`_event_table_columns:1351-1418` and
  `_contfrag_event_table_columns:862-909`) with the identical new trajectory
  columns, in the same order, so the parity test passes.
- **Docstrings in domain terms only** — no "Phase 2"/plan references in
  `trajectory.py`, column docs, or tests.

## Deliberately not in this phase

- New per-timebin signals — those are Phase 1. Phase 2 only reads Tier-1
  columns.
- Sequence/replay score, `radius_of_gyration`, shuffle significance,
  experienced-vs-shortcut path — deferred (spec "Considered and deferred").
- ContFrag-only `dominant_state` / `continuous_fraction` semantics — unchanged.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_trajectory_dynamics_hand_computed` | on a synthetic 2-event in-event frame, `path_length` = Σ step_distance, `net_displacement` = graph dist of endpoints, `tortuosity`, `n_reversals`, `pause_fraction`, `n_patches_visited`, `crosses_trunk`, `terminates_at_well` all match hand-computed values. |
| `test_ahead_behind_summary` | `ab_slope` sign matches a constructed outbound vs returning trajectory; `max_ahead`/`max_behind`/`remote_fraction` correct. |
| `test_brain_state_means_sum_to_one` | the four `*_prob_mean` columns sum to ≈1 per event on a normalized synthetic posterior. |
| `test_uncertainty_extra_multimodal` | a posterior split across two non-adjacent arms → `multimodal_fraction == 1`; single-arm → `0`. |
| `test_prev_arm_mirror` | `ends_at_prev` true when the trajectory ends at the just-taken well; mirrors `ends_at_next` logic. |
| `test_event_table_has_trajectory_columns` | `build_event_table` output contains every new column with expected dtype. |
| `test_contfrag_table_has_trajectory_columns` | same for `build_contfrag_event_table`. |
| `test_cross_scenario_parity` | the trajectory-metric column **names and dtypes** in the two tables are identical and produced by the same `trajectory.py` functions (guards against drift). |
| `test_path_continuity_merged` / `test_retrospective_merged` | the previously-standalone columns are present in the canonical event table. |

## Fixtures

- Extend `_tiny_dot_with_two_events` (`tests/unit/test_replay_tables.py:71-99`)
  to carry the Phase-1 per-bin columns (`step_distance`, raw ripple/MUA/theta,
  the four state probs, `run_speed_baseline`) so the rollup tests run on the
  existing synthetic `dot`.
- Reuse `_EnvStub` / `_bandit_graph` for graph-distance helpers, and the
  `test_retrospective.py` path fixtures for the `prev_arm` mirror.
- No real-data run required in Phase 2 (rollups are deterministic functions of
  Phase-1 columns); a single real-session sanity check that the table builds
  without error is optional but recommended.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Both tables get the identical trajectory column set from the shared `trajectory.py`; `test_cross_scenario_parity` passes.
- `_summarize_event_state`'s replaced responsibilities are fully covered by `brain_state_means` (no silent loss of `continuous_prob_mean`/`fragmented_prob_mean` semantics); the old code path is removed, not left parallel.
- The standalone retrospective/path-continuity call site replaced by the merge is named and removed (no orphan second pass).
- Tests are non-trivial (hand-computed expected values, not tautologies); shared setup is in fixtures.
- No docstring, test, or module name references this plan or "Phase 2".
